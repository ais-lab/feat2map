#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 13:35:47 2022

@author: thuan
"""
from demo_utils import VideoStreamer, make_plot_fast, PnP_Pose, Timer, GroundTruth, DSACresults
import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))
from utils.select_model import select_model
import torch
from models.trainer import load_state_dict
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2
from superpoint import SuperPoint
import pandas as pd
import os 
import time 
from visualize_utils import Model3D
import copy

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)

conf = {
        'grayscale': True,
        'resize_max': 640,
    }

dataset = "7scenes" 
scene = "chess"
config = "configsV"
scene_model = 2
is_plot_3d = False
thresholds = 0.5
root_src = os.path.dirname(os.getcwd())
# image_dataset_dir = "/media/thuan/8tb/Entire_HLOC/Hierarchical_Localization/datasets/"
image_dataset_dir = os.path.join(root_src, "/third_party/Hierarchical_Localization/datasets/")
###########################################################################################################

superpoint_conf = {
    'nms_radius': 3,
    'keypoint_threshold': 0.00,
    'max_keypoints': 2048,
    }
model, model_name = select_model(scene_model)
scene_checkpoint = "weights/chess.pth.tar"
print(scene_checkpoint)
####################  end initialization

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cuda_availale = True if device == 'cuda' else False
extractor_model = SuperPoint(superpoint_conf).eval().to(device)

# load the model
assert osp.isfile(scene_checkpoint)
loc_func = None if cuda_availale else lambda storage, loc: storage
checkpoint = torch.load(scene_checkpoint, map_location=loc_func)
load_state_dict(model, checkpoint['model_state_dict'])
model.eval().to(device)

img_path = dataset + "/" + scene + "/seq-03"
test_image_path = image_dataset_dir + img_path


glob = ['*.color.png'] if dataset == "7scenes" else ['*.png']
re_size = [1024, 576] 
vs = VideoStreamer(test_image_path, re_size, 1, glob, 1000000)
original_size = np.array([1920, 1080])
cam_scale = 0.2
time_measure = Timer()
ransac_pnp = PnP_Pose(dataset=dataset)

if is_plot_3d:
    model3d = Model3D(ransac_pnp.camera, camera_scale = cam_scale)
    model3d.create_window()
gt_pose = GroundTruth(root_src, dataset, scene)

dsacresults = DSACresults()

i = 0
while True:
    time_measure.start()
    frame, rgb_frame, image_file, ret = vs.next_frame()
    if frame is None and ret is False:
        print("------------------- END ---------------")
        if is_plot_3d: model3d.show()

    assert ret, 'Error when reading the first frame (try different --input?)'

    frame_tensor = frame2tensor(frame, device)
    data = {'image': frame_tensor}
    pred = extractor_model(data)
    scenes_pred = model({"descriptors":  torch.unsqueeze(pred['descriptors'][0], dim=0)})

    
    pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
    
    original_point2Ds = copy.deepcopy(pred['keypoints'])
    
    if dataset is not "7scenes":
        size = np.array(data['image'].shape[-2:][::-1])
        scales = (original_size / size).astype(np.float32)
        pred['keypoints'] = (pred['keypoints'] + .5) * scales[None] - .5
    
    points2D = pred["keypoints"]
    points3D = np.squeeze(scenes_pred[0].detach().cpu().numpy()).T
    uncertainty = np.squeeze(scenes_pred[1].detach().cpu().numpy()).T
    uncertainty = [True if tmpc > thresholds else False for tmpc in uncertainty]
    points3D = points3D[uncertainty,:]
    points2D = points2D[uncertainty,:]
    original_point2Ds_rgb = original_point2Ds[uncertainty,:]

    num_feats,_ = points2D.shape

    pose, ninliers = ransac_pnp.pnp(points2D, points3D)
    print("Re-Localizing image_file: ", image_file)
    print(f"Predicted Pose x:{pose[0][0]}, y:{pose[0][1]}, z:{pose[0][2]},\n qw:{pose[0][3]}, qx:{pose[0][4]}, qy:{pose[0][5]}, qz:{pose[0][6]} \n") 
    
    RGBs = [rgb_frame[int(i[1]),int(i[0]),:]/255 for i in original_point2Ds_rgb]
    
    if is_plot_3d: 
        model3d.add_points(points3D, RGBs)
        gtpose, status = gt_pose.get_gt_pose(image_file)
        if not status:
            continue
        dsac_pose, _ = dsacresults.get_pose(image_file)
        model3d.add_camera(pose)
        model3d.add_camera(gtpose, gt=True)
        model3d.add_camera(dsac_pose, othermethod=True)
        out = make_plot_fast(rgb_frame, original_point2Ds, uncertainty)
        cv2.imshow("test", out)
        key = chr(cv2.waitKey(1) & 0xFF)
        if key == 'p':
            # vs.cleanup()
            print('----------- PAUSE --------------')
            model3d.show()
            while True:
                key = chr(cv2.waitKey(1) & 0xFF)
                if key == 'c':
                    model3d = Model3D(ransac_pnp.camera, camera_scale = cam_scale)
                    model3d.create_window()
                    break
                elif key == 'q':
                    raise
        elif key == 'q':
            print('----------- STOP --------------')
            break
            
        if i == 0:
            model3d.show()
            # ---- for stop at the begining
            model3d = Model3D(ransac_pnp.camera, camera_scale = cam_scale)
            model3d.create_window()
        i += 1

        time_measure.infor(num_feats, ninliers)
        # time_measure.infor(0, 0)
if is_plot_3d: 
    cv2.destroyAllWindows()
    vs.cleanup()