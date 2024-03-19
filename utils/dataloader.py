import numpy as np
import os

def load_data(args):
    basedir = os.path.join(args.basedir, args.dataset)
    vt = np.load(os.path.join(basedir, 'v.npy')).astype(float)
    wt = np.load(os.path.join(basedir, 'w.npy')).astype(float)
    imu_ts = np.load(os.path.join(basedir, 't.npy')).astype(float)
    zts = np.load(os.path.join(basedir, 'zt.npy')).astype(float)
    Ks = np.load(os.path.join(basedir, 'Ks.npy')).astype(float)
    cam_T_imu = np.load(os.path.join(basedir, 'cam_T_imu.npy')).astype(float)
    K = np.load(os.path.join(basedir, 'K.npy')).astype(float)
    b = np.load(os.path.join(basedir, 'b.npy')).astype(float)

    return vt,wt,imu_ts,zts,Ks,cam_T_imu, K, b