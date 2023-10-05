#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 13:44:55 2022

@author: thuan
"""


import torch.nn as nn 
import torch 

def qvec2rotmat(qvec):
    return torch.tensor([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]]).to(device="cuda")

def project_loss(gt_pt2Ds, pt3Ds, c_pose, camera, valids):
    R = qvec2rotmat(c_pose[3:])
    t = torch.unsqueeze(c_pose[:3], dim = 1)
    f = camera[2]
    ppx = camera[3]
    ppy = camera[4]

    prd_2Ds = R@pt3Ds + t
    # project
    px = f*prd_2Ds[0,:]/prd_2Ds[2,:] + ppx
    py = f*prd_2Ds[1,:]/prd_2Ds[2,:] + ppy

    errors_x = (gt_pt2Ds[:,0] - px)**2
    errors_y = (gt_pt2Ds[:,1] - py)**2

    return torch.mean(valids * torch.sqrt(errors_x + errors_y))


class _3DFLCriterion(nn.Module):
    def __init__(self):
        super(_3DFLCriterion, self).__init__()
    
    def forward(self, coord_pd, coord_gt_dic, epoch):
        batch_size, _, _ = coord_pd[0].shape
        valids = coord_gt_dic["p3D_ids"]
        square_errors = torch.norm((coord_pd[0] - coord_gt_dic["p3Ds"]), dim = 1)
        l1 = torch.sum(valids*square_errors)/batch_size
        uncertainty_loss = torch.sum(torch.norm(valids - coord_pd[1]))/batch_size
    
        l2 = 0
        for i in range(batch_size):
                l2 += project_loss(coord_gt_dic['xys'][i,:,:], coord_pd[0][i,:,:], 
                                   coord_gt_dic['pose'][i,:], coord_gt_dic['camera'][i,:], valids[i,:])
        l2 = l2/batch_size
        
        if epoch >= 500:
            return l1+uncertainty_loss+10*l2, l1, l2, uncertainty_loss
        else:
            return l1+uncertainty_loss, l1, l2, uncertainty_loss

class fakezero(object):
	def __init__(self):
		pass
	def item(self):
		return 0
		

class _3DFLCriterion_New(nn.Module):
    def __init__(self, configs ={"coef_lrepr": 10, "start_lrepr": 10}):
        super(_3DFLCriterion_New, self).__init__()
        self.fakezero = fakezero()
        self.start_use_lr = configs['start_lrepr'] # start use loss reprojection 
        self.coef_lr = configs['coef_lrepr']
        print("----------- Reprojection Loss Setting:, start_lr: {}, coef: {}".format(self.start_use_lr, self.coef_lr))

    
    def forward(self, coord_pd, coord_gt_dic, epoch):
        batch_size, _, _ = coord_pd[0].shape
        valids = coord_gt_dic["p3D_ids"]
        square_errors = torch.norm((coord_pd[0] - coord_gt_dic["p3Ds"]), dim = 1)
        l1 = torch.sum(valids*square_errors)/batch_size
        uncertainty_loss = torch.sum(torch.norm(valids - coord_pd[1]))/batch_size
    
        l2 = 0
        count = 0
        for i in range(batch_size):
            if coord_gt_dic["is_real_label"][i] != 0:
                l2 += project_loss(coord_gt_dic['xys'][i,:,:], coord_pd[0][i,:,:], 
                                   coord_gt_dic['pose'][i,:], coord_gt_dic['camera'][i,:], valids[i,:])
                count += 1
        if count != 0:
            l2 = l2/count
            if epoch >= self.start_use_lr:
                return l1+uncertainty_loss + self.coef_lr*l2, l1, l2, uncertainty_loss
            else:
                return l1+uncertainty_loss, l1, l2, uncertainty_loss
        else:
            if epoch >= self.start_use_lr:
                return l1+uncertainty_loss + self.coef_lr*l2, l1, self.fakezero, uncertainty_loss
            else:
                return l1+uncertainty_loss, l1, self.fakezero, uncertainty_loss

    

def test_3DFLCriterion():
    # for testing
    import numpy as np 
    indata = np.array([[1,2,3,0],[1,2,4,0], [1,2,5,0], [1,2,6,0]])
    target = np.array([[1,2,2], [1,2,2], [1,2,2], [1,2,2]])
    valid_ids = np.array([1,1,1,1])
    
    indata = torch.unsqueeze(torch.from_numpy(indata.T).float(), dim=0)
    target = torch.unsqueeze(torch.from_numpy(target.T).float(), dim=0)
    valid_ids = torch.unsqueeze(torch.from_numpy(valid_ids).float(), dim=0)
    
    print("indata.shape: ", indata.shape)
    print(indata)
    print("target.shape: ", target.shape)
    print(target)
    print("valid_ids.shape: ", valid_ids.shape)
    
    target = {"p3Ds":target, "p3D_ids":valid_ids}
    loss = _3DFLCriterion() 
    print(loss(indata, target, 1))

if __name__ == "__main__":
    test_3DFLCriterion()
