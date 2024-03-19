import numpy as np
from pr3_utils import *


if __name__ == '__main__':

	# Load the measurements
	filename = "data/03.npz"
	t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
	Ks = np.zeros([4,4])
	Ks[:2,:3] = K[:2,:3]
	Ks[2:,:3] = K[:2,:3]
	fsu = K[0][0]
	Ks[2][3] = -fsu*b
	cam_T_imu = np.linalg.inv(imu_T_cam)
	linear_velocity = np.transpose(linear_velocity)
	angular_velocity = np.transpose(angular_velocity)
	features = features.transpose(2,1,0)
	t = np.squeeze(t)

	np.save('data/03/v.npy', linear_velocity)
	np.save('data/03/w.npy', angular_velocity)
	np.save('data/03/t.npy', t)
	np.save('data/03/Ks.npy', Ks)
	np.save('data/03/cam_T_imu.npy', cam_T_imu)
	np.save('data/03/zt.npy', features)
	np.save('data/03/K.npy', K)
	np.save('data/03/b.npy', b)

	# (a) IMU Localization via EKF Prediction

	# (b) Landmark Mapping via EKF Update

	# (c) Visual-Inertial SLAM

	# You may use the function below to visualize the robot pose over time
	# visualize_trajectory_2d(world_T_imu, show_ori = True)


