#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 19:00:05 2022

@author: thuan
"""

import os
import sys
import os.path as osp
sys.path.append(osp.join(osp.dirname(__file__), ".."))
import torch
from processing.Logger import History, Logger
from .trainer import step_fwd, load_state_dict
from tqdm import tqdm
from utils.vis_tool import PointCloud, Poses, evaluate_poses
import numpy as np
import pycolmap
import time

class Timer:
    def __init__(self, subject = "RANSAC PnP"):
        self.list_time = []
        self.subject = subject
    def start(self):
        self.start_time = time.time()
    def stop(self):
        self.list_time.append(time.time() - self.start_time)
    def eval(self):
        print("[INFOR] Average {} time {:.4f}s".format(self.subject, sum(self.list_time)/len(self.list_time)))


def qexp(q):
    n = np.linalg.norm(q)
    q = np.hstack((np.cos(n), np.sinc(n/np.pi)*q))
    return q


def PnP_Pose(points2D, points3D, npcamera, max_error_px=12.0):

    camera = {'model': 'SIMPLE_PINHOLE', 
              'width': npcamera[0], 'height': npcamera[1], 
              'params': npcamera[2:]} 
 
    ans = pycolmap.absolute_pose_estimation(points2D, points3D, camera, max_error_px)
    pose = np.zeros((1,7))
    pose[0,:3] = ans['tvec']
    pose[0,3:] = ans['qvec']
    return pose, ans['num_inliers']


class Evaluator(object):
    def __init__(self, experiment_name, model, test_criterion, checkpoint_file, test_set, 
                 train_set, data_path, train_criterion=None, dataset="7scenes",
                 eval_train=False, w_uncertainty={'state':False, 'threshold':0.5}, model_ver=1, folder_name = "eval"):
        self.experiment_name = experiment_name
        self.model = model
        self.train_criterion = train_criterion
        self.data_path = data_path
        self.dataset = dataset
        self.eval_train = eval_train
        self.w_uncertainty = w_uncertainty
        self.model_ver = model_ver
        if model_ver == 14 or model_ver >= 140:
            self.model_siam = True
        else:
            self.model_siam = False
        if test_criterion == None:
            self.test_criterion = self.train_criterion
        else:
            self.test_criterion = test_criterion
        
        self.config = {}
        self.config['cuda'] = torch.cuda.is_available()
        
        self.logdir = osp.join(os.getcwd(), 'logs', self.experiment_name, folder_name)
        if not osp.isdir(self.logdir):
            os.makedirs(self.logdir)
        self.clouds_target_path = osp.join(self.logdir, "target")
        self.prd_train_path = osp.join(self.logdir, "prd_train")
        self.prd_test_path = osp.join(self.logdir, "prd_test")
        self.prd_test_path_reg = osp.join(self.logdir, "prd_test_reg")
        self.prd_train_path_reg = osp.join(self.logdir, "prd_train_reg")
        
        self.logging = Logger(osp.join(self.logdir, "log.txt"))
        sys.stdout = self.logging

        assert checkpoint_file != None
        assert osp.isfile(checkpoint_file)
        self.checkpoint_file = checkpoint_file

        # load the model
        loc_func = None if self.config['cuda'] else lambda storage, loc: storage
        checkpoint = torch.load(self.checkpoint_file, map_location=loc_func)
        load_state_dict(self.model, checkpoint['model_state_dict'])
        print('Loaded checkpoint {:s} epoch {:d}'.format(self.checkpoint_file, checkpoint['epoch']))
        
        # dataloader
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(test_set,batch_size=1, shuffle=False)
        
        # activate GPUs
        if self.config['cuda']:
            self.model.cuda()
            self.train_criterion.cuda()
            self.test_criterion.cuda()
            
    
    def evaler(self): 
        """
        Function that does the validation. 
        """
        # Validation 

        self.model.eval()
        # initialize 
        if self.eval_train:
            print("Predicting Train Data")
            val_loss = History()
            L2 = History()
            Lpose = History()
            target_points = PointCloud()
            predict_points = PointCloud()
            prd_poses_train = Poses()
            prd_poses_train_reg = Poses()
            

            # prediction -- training data
            for batch_idx, (data, target) in tqdm(enumerate(self.train_loader)):
                loss, out = step_fwd(data, self.model, self.config['cuda'], target, self.test_criterion, epoch=1)
                val_loss.update(loss[0])
                L2.update(loss[2])
                if self.model_siam:
                    Lpose.update(loss[4])
                
                # PnP
                points2D = np.squeeze(data["keypoints"].cpu().numpy()) + 0.5
                points3D = np.squeeze(out[0].cpu().numpy()).T
                camera = np.squeeze(target["camera"].cpu().numpy())
                
                pose, num_inliers = PnP_Pose(points2D, points3D, camera)
                
                # print(num_inliers)
                prd_poses_train.update(pose, num_inliers)
                # update predicted 3D points
                target_points.update(np.squeeze(target['p3Ds'].cpu().numpy()).T)
                predict_points.update(points3D)

                if self.model_siam:
                    pred_re_pose =  np.squeeze(out[2].cpu().numpy())
                    # normalize the predicted quaternions
                    q = qexp(pred_re_pose[3:])
                    pred_re_pose = np.hstack((pred_re_pose[:3], q))
                    prd_poses_train_reg.update(pred_re_pose[np.newaxis,], 0)
            
            
            # save
            target_points.save(self.clouds_target_path)
            predict_points.save(self.prd_train_path)
            prd_poses_train.save(self.prd_train_path)
            print ('Val {:s}: l1_3Dmap {:f}'.format(self.experiment_name, val_loss.average()))
            print ('Val {:s}: l2_Reproj {:f}'.format(self.experiment_name, L2.average()))
            if self.model_siam:
                prd_poses_train_reg.save(self.prd_train_path_reg)
                print ('Val {:s}: l4_pose {:f}'.format(self.experiment_name, Lpose.average()))
        
        
        # prediction -- testing data
        print("Predicting Test Data")
        prd_poses_test = Poses(True)
        prd_poses_test_reg = Poses()
        predict_test_points = PointCloud()
        start_time = time.time()
        pnp_time_test = Timer("RANSAC PnP")
        d2s_time_test = Timer("D2S")
        list_num_test_features = [] 
        
        for batch_idx, data in tqdm(enumerate(self.test_loader)):
            img_name = data.pop('image_name')[0]
            
            d2s_time_test.start()
            _, out = step_fwd(data, self.model, self.config['cuda'])
            d2s_time_test.stop()

            points2D = np.squeeze(data["keypoints"].cpu().numpy()) + 0.5
            points3D = np.squeeze(out[0].cpu().numpy()).T
            camera = np.squeeze(data["camera"].cpu().numpy())

            ## 
            if self.w_uncertainty['state']:
                uncertainty = np.squeeze(out[1].cpu().numpy()).T
                # uncertainty = 1/(1+100*np.abs(uncertainty))
                uncertainty = [True if tmpc > self.w_uncertainty['threshold'] else False for tmpc in uncertainty]
                points3D = points3D[uncertainty,:]
                points2D = points2D[uncertainty,:]
                list_num_test_features.append(points2D.shape[0])
            ## 
            
            pnp_time_test.start()
            pose, num_inliers = PnP_Pose(points2D, points3D, camera)
            pnp_time_test.stop()
            # End PnP 
            if self.model_siam:
                pred_re_pose =  np.squeeze(out[2].cpu().numpy())
                # normalize the predicted quaternions
                q = qexp(pred_re_pose[3:])
                pred_re_pose = np.hstack((pred_re_pose[:3], q))
                prd_poses_test_reg.update(pred_re_pose[np.newaxis,], 0)
            ############### UPDATE
            
            prd_poses_test.update(pose, num_inliers, img_name)
            predict_test_points.update(points3D)
        
        d2s_time_test.eval()
        pnp_time_test.eval()  

        print("Average number of fine keypoints (removed uncertainty ones): {}".format(np.mean(list_num_test_features)))
        
        predict_test_points.save(self.prd_test_path)
        prd_poses_test.save(self.prd_test_path)
        if self.model_siam:
            prd_poses_test_reg.save(self.prd_test_path_reg)
        
        # evaluate the predicted camera poses
        if self.eval_train:
            print("[INFOR] Eval on training set: ")
            evaluate_poses(osp.join(self.data_path, "train", "readme.txt"), 
                            self.prd_train_path+".txt", False)
        
        print("[INFOR] Eval on testing set: ")
        mea_t, med_R, ratio = evaluate_poses(osp.join(self.data_path, "test", "readme.txt"), 
                       self.prd_test_path+".txt", True)
        
        if self.model_siam:
            if self.eval_train:
                print("--- Training data (Regress_branch): ")
                evaluate_poses(osp.join(self.data_path, "train", "readme.txt"), 
                                self.prd_train_path_reg+".txt", False)

            print("--- Testing data (Regress_branch): ")
            mea_t_, med_R_, ratio_ = evaluate_poses(osp.join(self.data_path, "test", "readme.txt"), 
                           self.prd_test_path_reg+".txt", True)
        
        return mea_t, med_R, ratio
        

        
    