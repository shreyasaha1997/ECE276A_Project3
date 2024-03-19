from utils.dataloader import *
from utils.ekf import *
# from utils.visual_slam_no_correlation import *
from utils.visual_slam import *
from utils.visualisations import *
import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--basedir", type=str, default='data/', 
                        help='where to load the data from')
    parser.add_argument("--dataset", type=str, default='10', 
                        help='experiment name')
    return parser

if __name__=='__main__':
    parser = config_parser()
    args = parser.parse_args()
    vts,wts,imu_ts,zts,Ks,cam_T_imu,K,b = load_data(args)

    poses, landmarks = optimise_poses(vts,wts,imu_ts,zts,cam_T_imu, Ks,K,b) ##uncomment this to get poses and landmarks via dead reckoning
    basedir = os.path.join(args.basedir, args.dataset)
    np.save(basedir + '/poses.npy', poses)
    np.save(basedir + '/landmarks.npy', landmarks)
    # landmarks = np.load(basedir + '/landmarks.npy')
    # poses = np.load(basedir + '/poses.npy')
    # poses, landmarks = visual_slam(args, vts,wts,imu_ts,zts,cam_T_imu, Ks,K,b)
    # np.save(basedir + '/poses_vs.npy', poses)
    # np.save(basedir + '/landmarks_vs.npy', landmarks)
    # visualise_landmarks(landmarks,poses.transpose(1,2,0), basedir + '/ekf_poses_landmarks.png',args)
    # visualize_trajectory_2d(poses.transpose(1,2,0),path_name="Unknown")