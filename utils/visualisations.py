import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler

def visualise_landmarks1(landmarks,pose,path,heading):
    Xs = landmarks[:,0]
    Ys = landmarks[:,1]
    plt.scatter(Xs, Ys, s=1)
    plt.plot(pose[0,3,:],pose[1,3,:],'r-')
    plt.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
    plt.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title(heading)
    plt.savefig(path)
    plt.close()

def visualise_landmarks(landmarks,pose,path,args):
    Xs = landmarks[:,0]
    Ys = landmarks[:,1]
    plt.scatter(Xs, Ys, s=1)
    plt.plot(pose[0,3,:],pose[1,3,:],'r-')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('initial poses and landmark estimates with dead reckoning and visual mapping for dataset ' + args.dataset)
    plt.savefig(path)

def visualize_trajectory_2d(pose,path_name="Unknown",show_ori=False):
    '''
    function to visualize the trajectory in 2D
    Input:
        pose:   4*4*N matrix representing the camera pose, 
                where N is the number of poses, and each
                4*4 matrix is in SE(3)
    '''
    fig,ax = plt.subplots(figsize=(5,5))
    n_pose = pose.shape[2]
    ax.plot(pose[0,3,:],pose[1,3,:],'r-',label=path_name)
    ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
    ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
  
    if show_ori:
        select_ori_index = list(range(0,n_pose,max(int(n_pose/50), 1)))
        yaw_list = []
        
        for i in select_ori_index:
            _,_,yaw = mat2euler(pose[:3,:3,i])
            yaw_list.append(yaw)
    
        dx = np.cos(yaw_list)
        dy = np.sin(yaw_list)
        dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
        ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
            color="b",units="xy",width=1)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.grid(False)
    ax.legend()
    plt.show(block=True)

    return fig, ax