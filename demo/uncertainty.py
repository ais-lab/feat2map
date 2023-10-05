#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 13:35:47 2022

@author: thuan
"""
import demo
import os.path as osp
from utils.select_model import select_model
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))
import torch
from models.trainer import load_state_dict
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2
from superpoint import SuperPoint
import pandas as pd
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from third_party.r2d2 import R2D2
import time

# raise

class Timer:
    def __init__(self, subject = "RANSAC PnP"):
        self.list_time = []
        self.subject = subject
    def start(self):
        self.start_time = time.time()
    def stop(self):
        self.list_time.append(time.time() - self.start_time)
    def eval(self):
        print("------- mean of {} is {} --------".format(self.subject, sum(self.list_time)/len(self.list_time)))

if __name__ == "__main__": 
    conf = {
            'grayscale': True,
            'resize_max': 640,
        }
    dataset = "Cambridge"
    scene = "KingsCollege"
    config = "configsV173"
    scene_model = 25
    is_plot = True
    train_data = False
    unlabel_data = False
    thresholds = [0.5, 0.6]
    ###########################################################################################################
    root = "/home/thuan/Desktop/GITHUB/FeatLoc_preparation/dataset/Hierarchical_Localization/datasets"
    infor_path = "/home/thuan/Desktop/GITHUB/feat2map/dataset"
    root_src = "/home/thuan/Desktop/GITHUB/feat2map"
    if dataset == "7scenes" or dataset == "Cambridge" or dataset == "BKC" :
        root = osp.join(root, dataset, scene)
        infor_path = osp.join(infor_path, dataset, scene)
    elif dataset == "12scenes":
        root = osp.join(root, dataset,"12scenes_sfm_triangulated" ,scene)
        infor_path = osp.join(infor_path, dataset, scene)
    elif dataset == "indoor6":
        root = osp.join(root, dataset, "indoor6_sfm_triangulated" ,scene)
        infor_path = osp.join(infor_path, dataset, scene)
    

    
    superpoint_conf = {
        'nms_radius': 3,
        'keypoint_threshold': 0.00,
        'max_keypoints': 2048,
        }
    model, model_name = select_model(scene_model)
        
    scene_checkpoint = osp.join(root_src, 'logs', 
                                dataset +'_'+ scene +'_'+config +'_'+model_name, "epoch_1640.pth.tar") # 6026 5580
    print(scene_checkpoint)
    ####################  end initialization
    
    
    if dataset == "12scenes":
        loader = demo.ImageDataset(root, infor_path, conf, dataset, is_plot)
    else:
        loader = demo.ImageDataset(root, infor_path, conf, train_data=train_data, unlabel_data = unlabel_data)

    # print(loader[0])
    # raise

    loader = torch.utils.data.DataLoader(loader, num_workers=0, batch_size=1, shuffle=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # device = 'cpu'

    cuda_availale = True if device == 'cuda' else False
    extractor_model = SuperPoint(superpoint_conf).eval().to(device)
    
    # extractor_model = R2D2(superpoint_conf).eval().to(device)
    
    # load the model
    assert osp.isfile(scene_checkpoint)
    loc_func = None if cuda_availale else lambda storage, loc: storage
    checkpoint = torch.load(scene_checkpoint, map_location=loc_func)
    load_state_dict(model, checkpoint['model_state_dict'])
    
    model.eval().to(device)

        
    figure, plot = plt.subplots(2, 2)
    figure.set_figheight(10)
    figure.set_figwidth(15)
    SuperPoint_timer = Timer("superpoint")
    D2S_timer = Timer("D2S")
    
    i = 0
    for data in tqdm(loader):
        # if i > 92:
        #     raise
        # if i != 92:
        #     i += 1
        #     continue
        image = np.squeeze(data['ori_image'].cpu().numpy())
        plot[0,0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plot[0,1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        plot[1,0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plot[1,1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        SuperPoint_timer.start()
        pred = extractor_model(demo.map_tensor(data, lambda x: x.to(device)))
        SuperPoint_timer.stop()
        # newdata = {'keypoints': data['keypoints'], 'descriptors': data['descriptors']}
        # pred = demo.map_tensor(newdata, lambda x: x.to(device))
        # raise
        D2S_timer.start()
        scenes_pred = model({"descriptors":  torch.unsqueeze(pred['descriptors'][0], dim=0)})
        D2S_timer.stop()
        
        
        pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
        
        pred['image_size'] = original_size = data['original_size'][0].cpu().numpy() 
        if 'keypoints' in pred:
            size = np.array(data['image'].shape[-2:][::-1])
            scales = (original_size / size).astype(np.float32)
            pred['keypoints'] = (pred['keypoints'] + .5) * scales[None] - .5
        
       
        points2D = pred["keypoints"]
        points2D_plot = points2D - 0.5
        
        o_uncertainty = np.squeeze(scenes_pred[1].detach().cpu().numpy())
        # o_uncertainty = 1/(1+100*np.abs(o_uncertainty))
        
        plot[0,0].scatter(points2D_plot[:,0], points2D_plot[:,1], color='lime', marker='o', s = 1)
        
        if train_data:
            data['xys'] = np.squeeze(data['xys'].detach().cpu().numpy()) - 0.5
            # print(data['xys'].shape)
            # print(points2D_plot.shape)
            # raise
            # plot[1,0].scatter(data['xys'][:,0], data["xys"][:,1], color='red', marker='o', s = 1)
            gt_uncertainty = np.squeeze(data['p3D_ids'].detach().cpu().numpy())
            gt_uncertainty = ['lime' if tmpc == 1.0 else 'r' for tmpc in gt_uncertainty]
            plot[1,1].scatter(data['xys'][:,0], data["xys"][:,1], c=gt_uncertainty, marker='o', s = 1)
        else:
            uncertainty = ['lime' if tmpc > thresholds[1] else 'r' for tmpc in o_uncertainty]
            plot[1,1].scatter(points2D_plot[:,0], points2D_plot[:,1], c=uncertainty, marker='o', s = 1)
            
        
        # uncertainty[uncertainty>0.9] = 1
        # uncertainty[uncertainty<0.9] = 0
        uncertainty = ['lime' if tmpc > thresholds[0] else 'r' for tmpc in o_uncertainty]

        plot[0,1].scatter(points2D_plot[:,0], points2D_plot[:,1], c=uncertainty, marker='o', s = 1)
        # save infor 
        length = points2D_plot.shape[0]
        plot_information = pd.DataFrame(np.zeros((length+1, 3)))
        plot_information.iloc[1:,:2] = points2D_plot
        plot_information.iloc[1:,2] = uncertainty
        plot_information.iloc[0,0] = data["name"]
        plot_information.to_csv('test/'+str(i)+'.txt', header = None, index = False, sep =" ")
        
        ################## 
        plot[0,1].set_title(str(thresholds[0]), fontsize=16)
        plot[1,1].set_title(str(thresholds[1]), fontsize=16)
        figure.savefig('test/'+ str(i) +'.svg')
        plot[0,0].clear()
        plot[0,1].clear()
        plot[1,1].clear()
        plot[1,0].clear()
        i+=1
        # if i > 30:
        # break 
        
    SuperPoint_timer.eval()
    D2S_timer.eval()
    