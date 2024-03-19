import numpy as np
from utils.visualisations import *

poses = np.load('data/10/poses.npy')
landmarks = np.load('data/10/landmarks.npy')
visualise_landmarks1(landmarks,poses.transpose(1,2,0),'outputs/dr_10.png','Dead Reckoning for dataset 10')


poses = np.load('data/10/vs3_final/poses.npy')
landmarks = np.load('data/10/vs3_final/mu_lndm_t.npy')
visualise_landmarks1(landmarks,poses.transpose(1,2,0),'outputs/slam_results_10.png','Visual SLAM Trajectory and Landmarks for dataset 10')

poses = np.load('data/10/vs2_no_correlation_landmarks/poses.npy')
landmarks = np.load('data/10/vs2_no_correlation_landmarks/mu_lndm_t.npy')
visualise_landmarks1(landmarks,poses.transpose(1,2,0),'outputs/slam_results_no_correlation_10.png','Results without pose-landmark correlation for dataset 10')

poses = np.load('data/10/vs1_fixed_landmark_priors/poses.npy')
landmarks = np.load('data/10/vs1_fixed_landmark_priors/mu_lndm_t.npy')
visualise_landmarks1(landmarks,poses.transpose(1,2,0),'outputs/slam_fixed_landmarks_10.png','Results with fixed landmark priors for dataset 10')

poses = np.load('data/03/poses.npy')
landmarks = np.load('data/03/landmarks.npy')
visualise_landmarks1(landmarks,poses.transpose(1,2,0),'outputs/dr_03.png','Dead Reckoning for dataset 03')

poses = np.load('data/03/vs3_final/poses.npy')
landmarks = np.load('data/03/vs3_final/mu_lndm_t.npy')
visualise_landmarks1(landmarks,poses.transpose(1,2,0),'outputs/slam_results_03.png','Visual SLAM Trajectory and Landmarks for dataset 03')

poses = np.load('data/03/vs2_no_correlation/poses.npy')
landmarks = np.load('data/03/vs2_no_correlation/mu_lndm_t.npy')
visualise_landmarks1(landmarks,poses.transpose(1,2,0),'outputs/slam_results_no_correlation_03.png','Results without pose-landmark correlation for dataset 03')

poses = np.load('data/03/vs1_fixed_landmark_priors/poses.npy')
landmarks = np.load('data/03/vs1_fixed_landmark_priors/mu_lndm_t.npy')
visualise_landmarks1(landmarks,poses.transpose(1,2,0),'outputs/slam_fixed_landmarks_03.png','Results with fixed landmark priors for dataset 03')