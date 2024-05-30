#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 13:29:13 2022

@author: thuan
"""

import os
import pandas as pd
import cv2
import numpy as np

def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, text, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text=[]):
    # this is original in https://github.com/magicleap/SuperGluePretrainedNetwork
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1
    out = np.stack([out]*3, -1)

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]

    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                  color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                    lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def makedir_OutScene(outdir, dataset ,scene):
    """
    Create a hierarchical folder for data preprocessing
    Parameters
    ----------
    outdir : path str
        Folder for generating train, test, val data.
        Ex: "/dataset/"
    dataset : str
        dataset name ex: "7Scenes"
        Ex: "/dataset/"
    scene : str
        scene name ex: "heads".
    Returns
    -------
    out_scene : str
        Return a path to created folders.
        Ex: /dataset/heads--train
                                --h5
                          --test
                                --h5
                          --val
                                --h5
    """
    out_scene = os.path.join(outdir,dataset)
    makedir(out_scene)
    out_scene = os.path.join(out_scene,scene)
    if os.path.exists(out_scene): #"The data path is existed"
        return None
    makedir(out_scene)
        
    list_out = ["train", "test", "unlabel", "augment", "val"]
    for i in list_out:
        out_i = os.path.join(out_scene,i)
        makedir(out_i)
        h5 = os.path.join(out_i, "h5")
        makedir(h5)
    return out_scene    
        

def numpy2str(array):
    out = ""
    for i in array:
        out += str(i) 
        out += " "
    return out.rstrip(out[-1])

    
def text_pose(tvec, qvec):
    """
    Generate a string list of 7DoF camera pose
    Parameters
    ----------
    tvec : numpy
        Translation vector.
    qvec : numpy
        Rotational vector.
    """
    out = numpy2str(tvec) + " " + numpy2str(qvec)
    return out

def cal_train_val(length:int, val_rate:float):
    """
    Calculate the size of train and validation.
    Parameters
    ----------
    length : int
        length of training data.
    val_rate : float
        rate of val data in training one.

    Returns
    -------
    list
        [train_size, test_size].

    """
    val_size = int(length*val_rate)
    train_size = length - val_size
    return [train_size, val_size]


def get_experiment_name(path):
    """
    Ex: 
        Input: "logs/7scenes_heads_configsV0/epoch_050.pth.tar"
        -> Output: 7scenes_heads_configsV0
    """
    out = ''
    s = False
    for i in path:
        if s:
            out = out + i
        if i == '/':
            s = ~s 
    return out.rstrip(out[-1])

def camera2txt(camera):
    """
    Ex: input a camera class of pycolmap type
    => output: a string of list params
    
    """
    out = str(camera.width) + " " + str(camera.height)
    for i in camera.params:
        out = out + " " + str(i)
    return out

class Cambridge_Cameras(object):
    def __init__(self, path):
        self.cameras = pd.read_csv(path, header = None, sep = " ")
        self.name2index = {str(self.cameras.iloc[i,0]):i for i in range(len(self.cameras))}
    def get_camera(self, name):
        out = self.cameras.iloc[self.name2index[name], 2:]
        return numpy2str(out.to_numpy())
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    