import numpy as np
from .matrix_utils import *

def motion_model(Tk, tao_k, wt, vt):

    zeta = np.array([vt[0],vt[1],vt[2],wt[0],wt[1],wt[2]])
    zeta = zeta
    zeta = np.expand_dims(zeta, axis=0)

    zeta_SE3 = np.squeeze(axangle2pose(zeta*tao_k))

    Tk1 = np.dot(Tk,zeta_SE3)
    return Tk1