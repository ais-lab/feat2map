#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 11:56:59 2022

@author: thuan
"""

import open3d as o3d
import numpy as np
import os.path as osp
import sys
import pandas as pd
import matplotlib.pyplot as plt
from .read_write_model import qvec2rotmat
sys.path.append(osp.join(osp.dirname(__file__), ".."))


class PointCloud(object):
    def __init__(self):
        self.data = None
    def update(self, new):
        _,d = new.shape
        assert d == 3
        if self.data is None:
            self.data = new 
        else:
            self.data = np.concatenate((self.data, new), axis = 0)
    def save(self, file):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.data)
        o3d.io.write_point_cloud(file +".ply", pcd)

class Poses(object):
    def __init__(self, test_data = False):
        self.data = None
        self.inliners = None
        self.names = None
        self.test_data = test_data
    def update(self, new, num_inliners, name = "None"):
        _,d = new.shape
        assert d == 7
        if self.data is None:
            self.data = new 
            self.inliners = [num_inliners]
            self.names = [name]
        else:
            self.data = np.concatenate((self.data, new), axis = 0)
            self.inliners.append(num_inliners)
            self.names.append(name)
    def save(self, file):
        print("Average number of inliers: {}".format(np.mean(self.inliners)))
        length = len(self.inliners)
        
        if self.test_data: # DSACSTAR eval
            pd_data = np.zeros((length, 8))
            pd_data = pd.DataFrame(pd_data)
            pd_data.iloc[:, 0] = self.names
            pd_data.iloc[:,1:5] = self.data[:,3:]
            pd_data.iloc[:,5:] = self.data[:,:3]
            pd_data.to_csv(file+'dsac.txt', header = False, sep = " ", index = False)
        # D2S eval
        pd_data = np.zeros((length, 8))
        pd_data = pd.DataFrame(pd_data)
        pd_data.iloc[:, 0] = self.inliners
        pd_data.iloc[:,1:] = self.data
        pd_data.to_csv(file+'.txt', header = False, sep = " ", index = False)
        
        

def vis_cloud(ply_file):
    pcd = o3d.io.read_point_cloud(ply_file)
    # pcd.paint_uniform_color([0.5,0.5,0.5])
    o3d.visualization.draw_geometries([pcd])



def plot_result(pred_poses, targ_poses):
    # thuan test
    pred_poses = pred_poses[:1000,:]
    targ_poses = targ_poses[:1000,:]
    # thuan test
    # this function is original from https://github.com/NVlabs/geomapnet
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    # plot on the figure object
    data_length = len(pred_poses)
    ss = max(1, int(data_length / 1000))  # 100 for stairs
    ss = 1
    # scatter the points and draw connecting line
    x = np.vstack((pred_poses[::ss, 0].T, targ_poses[::ss, 0].T))
    y = np.vstack((pred_poses[::ss, 1].T, targ_poses[::ss, 1].T))
    z = np.vstack((pred_poses[::ss, 2].T, targ_poses[::ss, 2].T))
    print("data.shape:", x.shape)
    for xx, yy, zz in zip(x.T, y.T, z.T):
      ax.plot(xx, yy, zs=zz, c='gray', alpha=0.6)
    ax.scatter(x[0, :], y[0, :], zs=z[0, :], s=5, c='r', depthshade=0, alpha=0.8)
    ax.scatter(x[1, :], y[1, :], zs=z[1, :], s=5, c='g', depthshade=0, alpha=0.8)
    ax.view_init(azim=119, elev=13)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    plt.show()


def get_listid_seq(pdas_data, seq = "seq-03"):
    indexes = [] 
    for i in range(len(pdas_data)):
        if seq in pdas_data.iloc[i,0]:
            indexes.append(i)
    
    return indexes
    


def evaluate_poses(gt_path, prd_path, mode=False, plot=False, seqid = None):
    if mode:
        # for testing data.
        indexes = [2, 5, 9]
    else:
        # for training data.
        indexes = [3, 6, 10]
    
    gt = pd.read_csv(gt_path, header=None, sep=" ")
    prd = pd.read_csv(prd_path, header=None, sep =" ")
    assert len(gt) == len(prd)

    errors_t = []
    errors_R = []
    for i in range(len(gt)):
        R_gt = qvec2rotmat(gt.iloc[i,indexes[1]:indexes[2]].to_numpy())
        
        t_gt = gt.iloc[i,indexes[0]:indexes[1]].to_numpy()
        
        t = prd.iloc[i,1:4].to_numpy()
       
        R = qvec2rotmat(prd.iloc[i,4:].to_numpy())

        e_t = np.linalg.norm(-R_gt.T @ t_gt + R.T @ t, axis=0)
        cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1., 1.)
        e_R = np.rad2deg(np.abs(np.arccos(cos)))
        errors_t.append(e_t)
        errors_R.append(e_R)

    errors_t = np.array(errors_t)
    errors_R = np.array(errors_R)

    med_t = np.median(errors_t)
    med_R = np.median(errors_R)

    print('Median errors: {:.4f}m, {:.4f}deg'.format(med_t, med_R))

    print('Percentage of test images localized within:')
    threshs_t = [0.01, 0.02, 0.03, 0.05, 0.1]
    threshs_R = [1.0, 2.0, 3.0, 5.0, 10.0]
    ratios = []
    for th_t, th_R in zip(threshs_t, threshs_R):
        ratio = np.mean((errors_t < th_t) & (errors_R < th_R))
        print('\t{:.0f}cm, {:.0f}deg : {:.2f}%'.format(th_t*100, th_R, ratio*100))
        #####
        if th_t >= 0.05:
            ratios.append(ratio)
    if plot:
        if seqid is not None:
            list_indexes = get_listid_seq(gt , seq = seqid)
            pred_poses = prd.iloc[:,1:].to_numpy()[list_indexes,:]
            targ_poses = gt.iloc[:,2:9].to_numpy()[list_indexes,:]
        else:
            pred_poses = prd.iloc[:,1:].to_numpy()
            targ_poses = gt.iloc[:,2:9].to_numpy()
        plot_result(pred_poses, targ_poses)
    return med_t, med_R, ratios
    




if __name__ == "__main__":
    # file = "/home/thuan/Desktop/GITHUB/ICRA23/logs/prediction.ply"
    # vis_cloud(file)
    gt = "/home/thuan/Desktop/GITHUB/ICRA23/dataset/7scenes/fire/test/readme.txt"
    prd = "/home/thuan/Desktop/GITHUB/ICRA23/logs/7scenes_fire_configsV0/eval/prd_test.txt"
    evaluate_poses(gt, prd, True)
