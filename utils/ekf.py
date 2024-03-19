import numpy as np
from scipy.linalg import expm
from .models import *
from .matrix_utils import *
from tqdm import tqdm

P = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
V = np.diag([0.01,0.01,0.01,0.01])
I = np.eye(3)

def initialize_priors():
    mu = np.eye(4)
    sigma = np.diag([0.001,0.001,0.001,0.0001,0.0001,0.0001])
    return mu,sigma

def only_prediction_step(mu,sigma,vts,wts,taos):
    zetas = np.concatenate((vts, wts), axis=1)
    zeta_tau = np.array([tau*z for tau,z in zip(taos,zetas)])
    zeta_tau_hat = axangle2pose(zeta_tau)
    zeta_tau_curly = expm(axangle2adtwist(zeta_tau*(-1.)))
    poses = [mu]
    sigmas = [sigma]
    for zth,ztc in zip(zeta_tau_hat,zeta_tau_curly):
        pose_k = np.matmul(poses[-1],zth)
        poses.append(pose_k)
        sigma_k = ztc@sigmas[-1]@ztc.T
        sigmas.append(sigma_k)
    return poses, sigmas

def initialize_landmark_priors_for_particular_pose(M,z,z1,z2,K,b):

    z1_h, z2_h = to_hom(z1), to_hom(z2)
    z1_n, z2_n = np.expand_dims(z1_h, axis=-1), np.expand_dims(z2_h, axis=-1) 
    K = np.tile(K, (M, 1, 1))
    z1_n, z2_n = np.linalg.inv(K)@z1_n, np.linalg.inv(K)@z2_n

    R = np.tile(I, (M, 1, 1))
    p, e3 = np.array([[b],[0],[0]]), np.array([[0],[0],[1]])
    p = np.tile(p, (M, 1, 1))
    e3 = np.tile(e3, (M, 1, 1))

    s1 = e3.transpose(0,2,1)@R.transpose(0,2,1)@p
    a = R.transpose(0,2,1)@p - z2_n*s1
    s2 = e3.transpose(0,2,1)@R.transpose(0,2,1)@z1_n
    b = R.transpose(0,2,1)@z1_n - z2_n*s2
    
    num = a.transpose(0,2,1)@a
    den = a.transpose(0,2,1)@b
    den = den+0.0000001
    mu = z1_n@(num/den)
    mu = np.squeeze(mu, axis=-1)

    cam_dist = np.array([np.linalg.norm(i) for i in mu])

    non_obs_indices = np.where(np.all(z == [-1, -1, -1, -1], axis=1))
    outlier_indices = np.where(z1[:, 0] < z2[:, 0])
    dist_outliers = np.where(cam_dist>100.)
    valid_indices = [i for i in range(M) if i not in non_obs_indices[0] and i not in outlier_indices[0] and i not in dist_outliers[0]]
    return mu,valid_indices

def initialize_landmark_priors(M, poses, zts, K, b, imu_T_cam):
    sigma = np.array([np.eye(3) for i in range(M)])
    mu = np.zeros([M,3])
    imu_T_cam = np.tile(imu_T_cam, (M, 1, 1))

    for pose,z in tqdm(zip(poses[:],zts[:])):
        z1 = z[:,:2]
        z2 = z[:,2:]
        mu_t,valid_indices = initialize_landmark_priors_for_particular_pose(M,z,z1,z2,K,b)
        w_T_imu = np.tile(pose, (M, 1, 1))
        mu_t = np.expand_dims(to_hom(mu_t), axis=-1)
        mu_t = w_T_imu@imu_T_cam@mu_t
        mu_t = np.squeeze(mu_t, axis=-1)
        for ind in valid_indices:
            mu[ind] = mu_t[ind][:3]
    return mu,sigma


def visual_mapping_update_step(mu_t, sigma_t, oTi, Tt1, Ks, zt):
    mu_t_hom = to_hom(mu_t)
    mu_t_hom = np.expand_dims(mu_t_hom, axis=-1)
    transform = oTi@np.linalg.inv(Tt1)
    transform = np.tile(transform, (mu_t_hom.shape[0], 1, 1))
    transformed_mu_t = transform@mu_t_hom
    transformed_mu_t = np.squeeze(transformed_mu_t, axis=-1)

    proj_jac_transformed_mu_t = projectionJacobian(transformed_mu_t)

    Ks = np.tile(Ks, (mu_t.shape[0], 1, 1))
    Ps = np.tile(P, (mu_t.shape[0], 1, 1))
    Vs = np.tile(V, (mu_t.shape[0], 1, 1))
    Is = np.tile(I, (mu_t.shape[0], 1, 1))

    Ht1 = Ks@proj_jac_transformed_mu_t@transform@Ps.transpose(0,2,1)

    Kt1 = Ht1@sigma_t@Ht1.transpose(0,2,1) + Vs
    Kt1 = np.linalg.inv(Kt1)
    Kt1 = sigma_t@Ht1.transpose(0,2,1)@Kt1

    pred_z = transform@mu_t_hom
    pred_z = np.squeeze(pred_z, axis=-1)
    pred_z = projection(pred_z)
    pred_z = np.expand_dims(pred_z, axis=-1)
    pred_z = Ks@pred_z
    pred_z = np.squeeze(pred_z, axis=-1)
    update = zt-pred_z
    update = np.expand_dims(update, axis=-1)
    update = np.squeeze(Kt1@update, axis=-1)
    indices = np.where(np.all(zt == [-1, -1, -1, -1], axis=1))
    update[indices] = [0,0,0]
    mu_t1 = mu_t + update

    sigma_t1 = (Is - Kt1@Ht1)@sigma_t

    return mu_t1, sigma_t1

def visual_landmark_mapping(zts, poses, oTi, Ks,K,b):
    M = zts.shape[1]

    mu,sigma = initialize_landmark_priors(M, poses, zts, K, b, np.linalg.inv(oTi))
    for pose,zt in tqdm(zip(poses,zts)):
        mu,sigma = visual_mapping_update_step(mu,sigma, oTi, pose, Ks, zt)
    return mu,sigma

def optimise_poses(vts,wts,imu_ts,zts,oTi, Ks,K,b):
    taos = [(imu_ts[i]-imu_ts[i-1]) for i in range(1,len(imu_ts))]
    taos.append(0)
    mu,sigma = initialize_priors()

    ## Prediction step
    mu_poses, sigmas_poses = only_prediction_step(mu,sigma,vts,wts,taos)

    ## Update Step
    mu_landmarks,sigma_landmarks = visual_landmark_mapping(zts,mu_poses[:-1],oTi, Ks,K,b)

    return np.array(mu_poses), np.array(mu_landmarks)