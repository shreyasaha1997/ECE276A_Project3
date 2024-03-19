from .matrix_utils import *
from .visualisations import *
from tqdm import tqdm
from scipy.linalg import expm
import numpy as np

I = np.eye(3)
I6 = np.eye(6)
P = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
V = np.diag([0.01,0.01,0.01,0.01])

def initialize_pose_priors():
    mu = np.eye(4)
    return mu

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

def initialize_landmark_priors(M,zt,K,b,imu_T_cam,pose,mu):
    imu_T_cam = np.tile(imu_T_cam, (M, 1, 1))
    z1 = zt[:,:2]
    z2 = zt[:,2:]
    mu_t,valid_indices = initialize_landmark_priors_for_particular_pose(M,zt,z1,z2,K,b)
    non_initiated_landmarks = np.where(np.all(mu == [0.,0.,0.], axis=1))
    valid_indices = np.array([ind for ind in valid_indices if ind in non_initiated_landmarks[0]])

    w_T_imu = np.tile(pose, (M, 1, 1))
    mu_t = np.expand_dims(to_hom(mu_t), axis=-1)
    mu_t = w_T_imu@imu_T_cam@mu_t
    mu_t = np.squeeze(mu_t, axis=-1)
    for ind in valid_indices:
        mu[ind] = mu_t[ind][:3]
    return mu

def predict(M, pose_t, sigma_t, vt, wt, tau):
    zeta = tau * np.concatenate((vt, wt), axis=0)
    zeta = np.expand_dims(zeta, axis=0)
    zeta_tau_hat = axangle2pose(zeta)
    zeta_tau_curly = expm(axangle2adtwist(zeta*(-1.)))
    zeta_tau_hat = np.squeeze(zeta_tau_hat, axis=0)
    zeta_tau_curly = np.squeeze(zeta_tau_curly, axis=0)
    pose_t1 = np.matmul(pose_t,zeta_tau_hat)

    sigma_lr = sigma_t[:3*M,3*M:]
    sigma_rl = sigma_t[3*M:,:3*M]
    sigma_rr = sigma_t[3*M:,3*M:]

    sigma_lr = sigma_lr@zeta_tau_curly.T
    sigma_rl = zeta_tau_curly@sigma_rl
    sigma_rr = zeta_tau_curly@sigma_rr@zeta_tau_curly.T

    sigma_t[:3*M,3*M:] = sigma_lr
    sigma_t[3*M:,:3*M] = sigma_rl
    sigma_t[3*M:,3*M:] = sigma_rr
    return pose_t1, sigma_t

def get_pose_Jacobian(Ks, m, oTi, mu_t1_t):
    m_hom = to_hom(m)
    m_hom = np.expand_dims(m_hom, axis=-1)

    transform = oTi@np.linalg.inv(mu_t1_t)
    transform = np.tile(transform, (m_hom.shape[0], 1, 1))
    transformed_mu_t = transform@m_hom
    transformed_mu_t = np.squeeze(transformed_mu_t, axis=-1)
    proj_jac_transformed_mu_t = projectionJacobian(transformed_mu_t)

    mu_t1_t = np.tile(mu_t1_t, (m_hom.shape[0], 1, 1))
    circle_fun = np.linalg.inv(mu_t1_t)@m_hom
    circle_fun = np.squeeze(circle_fun, axis=-1)[:,:3]
    circle_fun = circleDotFun(circle_fun)

    Ks = np.tile(Ks, (m_hom.shape[0], 1, 1))
    oTi = np.tile(oTi, (m_hom.shape[0], 1, 1))
    Ht1 = -Ks@proj_jac_transformed_mu_t@oTi@circle_fun

    return Ht1

def get_landmarks_Jacobian(Ks, mu_t, oTi, Tt1):

    mu_t_hom = to_hom(mu_t)
    mu_t_hom = np.expand_dims(mu_t_hom, axis=-1)
    transform = oTi@np.linalg.inv(Tt1)
    transform = np.tile(transform, (mu_t_hom.shape[0], 1, 1))
    transformed_mu_t = transform@mu_t_hom
    transformed_mu_t = np.squeeze(transformed_mu_t, axis=-1)

    proj_jac_transformed_mu_t = projectionJacobian(transformed_mu_t)

    Ps = np.tile(P, (mu_t.shape[0], 1, 1))
    Ks = np.tile(Ks, (mu_t.shape[0], 1, 1))
    Ht1 = Ks@proj_jac_transformed_mu_t@transform@Ps.transpose(0,2,1)

    return Ht1

def get_kalman_gain(Ht1, sigma_t1):
    Kg = Ht1@sigma_t1@Ht1.transpose(1,0)
    V = np.diag([0.01 for i in range(Kg.shape[0])])
    Kg = Kg+V
    Kg = np.linalg.inv(Kg)
    Kg = sigma_t1@Ht1.transpose(1,0)@Kg
    return Kg

def filter_sigma_matrix(M, sigma, valid_indices):
    valid_indices_sigma = []
    for ind in valid_indices:
        valid_indices_sigma.append(3*ind)
        valid_indices_sigma.append(3*ind+1)
        valid_indices_sigma.append(3*ind+2)
    valid_indices_sigma = np.array(valid_indices_sigma)

    sigma_ll = sigma[:3*M,:3*M]
    sigma_lr = sigma[:3*M,3*M:]
    sigma_rl = sigma[3*M:,:3*M]
    sigma_rr = sigma[3*M:,3*M:]

    sigma_ll_filtered = []
    for i in valid_indices_sigma:
        sigma_row = []
        for j in valid_indices_sigma:
            sigma_row.append(sigma[i][j])
        sigma_ll_filtered.append(sigma_row)
    sigma_ll_filtered = np.array(sigma_ll_filtered)

    sigma_lr_filtered = []
    for i in valid_indices_sigma:
        sigma_row = sigma[i][3*M:]
        sigma_lr_filtered.append(sigma_row)
    sigma_lr_filtered = np.array(sigma_lr_filtered)

    sigma_rl_filtered = []
    for i in range(3*M,3*M+6):
        sigma_row = sigma[i][valid_indices_sigma]
        sigma_rl_filtered.append(sigma_row)
    sigma_rl_filtered = np.array(sigma_rl_filtered)

    sigma_filtered = sigma_ll_filtered
    sigma_filtered = np.concatenate((sigma_filtered,sigma_lr_filtered),axis=1)
    den = np.concatenate((sigma_rl_filtered, sigma_rr), axis=1)
    sigma_filtered = np.concatenate((sigma_filtered,den),axis=0)
    return sigma_filtered

def get_obs_difference(zt, Ks, oTi, mu_t1_t_poses, mu_lndm_t):
    oTi = np.tile(oTi, (mu_lndm_t.shape[0], 1, 1))
    mu_t1_t_poses = np.tile(mu_t1_t_poses, (mu_lndm_t.shape[0], 1, 1))
    Ks = np.tile(Ks, (mu_lndm_t.shape[0], 1, 1))

    mu_lndm_t_hom = to_hom(mu_lndm_t)
    mu_lndm_t_hom = np.expand_dims(mu_lndm_t_hom, axis=-1)
    pred_z = oTi@np.linalg.inv(mu_t1_t_poses)@mu_lndm_t_hom
    pred_z = np.squeeze(pred_z, axis=-1)
    pred_z = projection(pred_z)
    pred_z = np.expand_dims(pred_z, axis=-1)
    pred_z = Ks@pred_z
    pred_z = np.squeeze(pred_z, axis=-1)
    return zt-pred_z

def update_mu_poses(valid_indices, mu_t1_t_poses, Kg_poses, obs_diff):
    mu_t1_t1_poses = obs_diff
    mu_t1_t1_poses = mu_t1_t1_poses[valid_indices]
    mu_t1_t1_poses = mu_t1_t1_poses.reshape(-1, 1)
    mu_t1_t1_poses = Kg_poses@mu_t1_t1_poses
    mu_t1_t1_poses = np.squeeze(mu_t1_t1_poses, axis=-1)
    mu_t1_t1_poses = np.expand_dims(mu_t1_t1_poses, axis=0)
    mu_t1_t1_poses = axangle2pose(mu_t1_t1_poses)
    mu_t1_t1_poses = np.squeeze(mu_t1_t1_poses, axis=0)
    mu_t1_t1_poses = mu_t1_t_poses@mu_t1_t1_poses
    return mu_t1_t1_poses

def update_mu_landmarks(valid_indices, mu_t1_t_poses,mu_lndm_t, Kg_landmarks, zt, Ks, oTi, obs_diff):
    mu_lndm_t1 = obs_diff
    mu_lndm_t1 = get_obs_difference(zt, Ks, oTi, mu_t1_t_poses, mu_lndm_t)
    mu_lndm_t1 = np.expand_dims(mu_lndm_t1, axis=-1)
    mu_lndm_t1 = mu_lndm_t1[valid_indices]
    mu_lndm_t1 = mu_lndm_t1.reshape(-1,1)
    mu_lndm_t1 = Kg_landmarks@mu_lndm_t1
    mu_lndm_t1 = np.squeeze(mu_lndm_t1, axis=-1)
    mu_lndm_t1 = np.reshape(mu_lndm_t1, (-1, 3))
    mu_lndm_t[valid_indices] = mu_lndm_t[valid_indices] + mu_lndm_t1
    return mu_lndm_t

def convert_filtered_sigma_to_whole_sigma(M, valid_indices, sigma_filtered, sigma_t):
    valid_indices_sigma = []
    for ind in valid_indices:
        valid_indices_sigma.append(3*ind)
        valid_indices_sigma.append(3*ind+1)
        valid_indices_sigma.append(3*ind+2)
    valid_indices_sigma = np.array(valid_indices_sigma)

    m = len(valid_indices)
    sigma_ll_filtered = sigma_filtered[:3*m,:3*m]
    sigma_lr_filtered = sigma_filtered[:3*m,3*m:]
    sigma_rl_filtered = sigma_filtered[3*m:,:3*m]
    sigma_rr_filtered = sigma_filtered[3*m:,3*m:]

    sigma_ll = sigma_t[:3*M,:3*M]
    sigma_lr = sigma_t[:3*M,3*M:]
    sigma_rl = sigma_t[3*M:,:3*M]
    sigma_rr = sigma_t[3*M:,3*M:]

    for a,i in enumerate(valid_indices_sigma):
        for b,j in enumerate(valid_indices_sigma):
            sigma_ll[i][j] = sigma_ll_filtered[a][b]

    for a,i in enumerate(valid_indices_sigma):
        sigma_lr[i] = sigma_lr_filtered[a]

    for a,i in enumerate(valid_indices_sigma):
        sigma_rl[0][i] = sigma_rl_filtered[0][a]
        sigma_rl[1][i] = sigma_rl_filtered[1][a]
        sigma_rl[2][i] = sigma_rl_filtered[2][a]
        sigma_rl[3][i] = sigma_rl_filtered[3][a]
        sigma_rl[4][i] = sigma_rl_filtered[4][a]
        sigma_rl[5][i] = sigma_rl_filtered[5][a]
    
    sigma_rr = sigma_rr_filtered

    sigma_t[:3*M,:3*M] = sigma_ll
    sigma_t[:3*M,3*M:] = sigma_lr
    sigma_t[3*M:,:3*M] = sigma_rl
    sigma_t[3*M:,3*M:] = sigma_rr
    return sigma_t

def update(M, Ks, mu_lndm_t, oTi, mu_t1_t_poses, sigma_t1_t,zt):
    indices = np.where(np.all(zt == [-1, -1, -1, -1], axis=1))
    valid_indices = np.array([i for i in range(mu_lndm_t.shape[0]) if i not in indices[0]])

    H_poses = get_pose_Jacobian(Ks, mu_lndm_t, oTi, mu_t1_t_poses)
    H_poses = H_poses[valid_indices]
    H_poses = H_poses.reshape(-1,6)

    H_landmarks = get_landmarks_Jacobian(Ks, mu_lndm_t, oTi, mu_t1_t_poses) 
    H_landmarks_extended = np.zeros([4*len(valid_indices),3*len(valid_indices)])
    for i,ind in enumerate(valid_indices):
        H_landmarks_extended[4*i:4*i+4,3*i:3*i+3] = H_landmarks[ind]
    H_landmarks_extended = np.array(H_landmarks_extended)

    H_t1 = np.concatenate((H_landmarks_extended, H_poses), axis=1)

    sigma_filtered = filter_sigma_matrix(M, sigma_t1_t, valid_indices)
    Kg = get_kalman_gain(H_t1, sigma_filtered)
    Kg_poses = Kg[3*len(valid_indices):,:]
    Kg_landmarks = Kg[:3*len(valid_indices),:]

    obs_diff = get_obs_difference(zt, Ks, oTi, mu_t1_t_poses, mu_lndm_t)
    mu_t1_t1_poses = update_mu_poses(valid_indices, mu_t1_t_poses, Kg_poses, obs_diff)
    mu_lndm_t1 = update_mu_landmarks(valid_indices, mu_t1_t_poses,mu_lndm_t, Kg_landmarks, zt, Ks, oTi, obs_diff)
    sigma_filtered_t1 = (np.eye(sigma_filtered.shape[0]) - Kg@H_t1)@sigma_filtered
    sigma_t1_t1 = convert_filtered_sigma_to_whole_sigma(M, valid_indices, sigma_filtered_t1, sigma_t1_t)

    return mu_t1_t1_poses, mu_lndm_t1, sigma_t1_t1

def visual_slam(args, vts,wts,imu_ts,zts,cam_T_imu, Ks,K,b):
    M = zts.shape[1]

    # initialise pose priors for pose zero
    # mu_pose_t = initialize_pose_priors()
    # mu_lndm_t = np.zeros([M,3])
    # mu_lndm_t = initialize_landmark_priors(M,zts[0],K,b,np.linalg.inv(cam_T_imu),mu_pose_t,mu_lndm_t)
    # sigma_t1_t = np.diag([1 for i in range(3*M+6)])
    # poses = [mu_pose_t]

    #load if already present
    mu_pose_t = np.load('data/10/vs3/poses.npy')
    mu_lndm_t = np.load('data/10/vs3/mu_lndm_t.npy')
    sigma_t1_t = np.load('data/10/vs3/sigma_lndm_t.npy')
    ##initialize landmark priors for pose 0
    mu_pose_t = mu_pose_t[:]
    poses = list(mu_pose_t)
    mu_pose_t = mu_pose_t[-1]

    for i in tqdm(range(2900,len(vts))):
        vt, wt, tau, zt = vts[i], wts[i], imu_ts[i]-imu_ts[i-1], zts[i]


        mu_t1_t, sigma_t1_t = predict(M, mu_pose_t, sigma_t1_t, vt, wt, tau)

        ##update priors for landmarks that haven't been see yet
        mu_lndm_t  = initialize_landmark_priors(M,zt,K,b,np.linalg.inv(cam_T_imu),mu_t1_t,mu_lndm_t)

        mu_t1_t1_poses, mu_lndm_t1, sigma_t1_t1 = update(M, Ks, mu_lndm_t, cam_T_imu, mu_t1_t, sigma_t1_t,zt)
        mu_pose_t, mu_lndm_t, sigma_t1_t = mu_t1_t1_poses, mu_lndm_t1, sigma_t1_t1
        poses.append(mu_pose_t)
    
    poses = np.array(poses)
    np.save('data/10/vs3/poses.npy', poses)
    np.save('data/10/vs3/mu_lndm_t.npy', mu_lndm_t)
    np.save('data/10/vs3/sigma_lndm_t.npy', sigma_t1_t)
    visualise_landmarks(mu_lndm_t,poses.transpose(1,2,0),'testing2.png',args)
    return mu_lndm_t,poses