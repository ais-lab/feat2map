#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 19:29:06 2022

@author: thuan
"""

from torch.utils.data import Dataset
import torch
import pandas as pd
import os.path as osp 
import h5py
import numpy as np 
import random
def qlog(q):
    if all(q[1:] == 0):
        q = np.zeros(3)
    else:
        q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
    return q


class _3DFeatLoc_Loader(Dataset):
    
    def __init__(self, data_dir, mode = "train", configs=dict):
        self.data_dir = osp.join(data_dir, mode)
        self.infor = pd.read_csv(osp.join(self.data_dir, "readme.txt"), header=None, sep =" ")
        self.num_feats = 2048
        self.configs = configs 
        
    def __len__(self):
        return len(self.infor)
    
    def __getitem__(self, idx):
        self.num_feats = 2048
        # input 
        name_train = self.infor.iloc[idx,1]
        h5File_train = osp.join(self.data_dir, "h5", name_train)
        features_train = h5py.File(h5File_train, 'r')
        data = {}
        for k,v in features_train[name_train.replace(".h5", "")].items():    
            if k == 'image_size':
                data[k] = torch.from_numpy(v.__array__()).float()
            elif  k == 'descriptors':
                data[k] = torch.from_numpy(v.__array__()[:,:self.num_feats]).float()
            else:
                data[k] = torch.from_numpy(v.__array__()[:self.num_feats]).float()
                # print(data[k].shape, k)
            # data[k] = torch.from_numpy(v.__array__()).float()
        camera = self.infor.iloc[idx, 10:].to_numpy().astype(float)
        pose = self.infor.iloc[idx, 3:10].to_numpy().astype(float)
        
        # target
        name_target = self.infor.iloc[idx,2]
        h5File_target = osp.join(self.data_dir, "h5", name_target)
        features_target = h5py.File(h5File_target, 'r')
        target = {}
        target["camera"] = torch.from_numpy(camera).float()
        target["pose"] = torch.from_numpy(pose).float()
        tmp_pose = np.zeros(6)
        tmp_pose[:3] = pose[:3]
        tmp_pose[3:] = qlog(pose[3:])
        target["pose_nor"] = torch.from_numpy(tmp_pose).float()
        for k,v in features_target[name_target.replace(".h5", "")].items(): 
            if k =="p3D_ids":
                v = v.__array__()
                v[v!=-1] = 1
                v[v==-1] = 0
                target[k] = torch.from_numpy(v[:self.num_feats]).float()
            elif k == "p3Ds": 
                target[k] = torch.from_numpy(v.__array__()[:self.num_feats].T).float() 
            else:
                target[k] = torch.from_numpy(v.__array__()[:self.num_feats]).float()
            
        return data, target


############################# 
class _3DFeatLoc_Loader_New(Dataset):
    
    def __init__(self, data_dir, mode = "train", configs={"unlabel":False, "augment":False}):
        self.data_dir = osp.join(data_dir, mode)
        self.infor = pd.read_csv(osp.join(self.data_dir, "readme.txt"), header=None, sep =" ")
        self.length_train = len(self.infor)
        self.num_feats = 2048
        self.configs = configs 
        if self.configs["augment"]:
            self.unlabel_data_dir = osp.join(data_dir, "augment")
            self.unlabel_infor = pd.read_csv(osp.join(self.unlabel_data_dir, "readme.txt"), header=None, sep=" ")
            self.length_unlabel = len(self.unlabel_infor)
            self.list_unlabel_indexes = list(range(self.length_unlabel))
            random.shuffle(self.list_unlabel_indexes)
            self.list_unlabel_indexes = self.list_unlabel_indexes[:int(self.configs['unlabel_rate']*self.length_unlabel)]
            self.length_unlabel = len(self.list_unlabel_indexes)
            self.fake_camera = self.infor.iloc[0, 10:].to_numpy().astype(float) # this is not correct, please fix this for future use
            
        
    def __len__(self):
        if self.configs["augment"]:
            return self.length_train + self.length_unlabel
        else:
            return self.length_train
    
    def __getitem__(self, idx):
        self.num_feats = 2048
        # input 
        if idx < self.length_train:
            name_train = self.infor.iloc[idx,1]
            h5File_train = osp.join(self.data_dir, "h5", name_train)
            features_train = h5py.File(h5File_train, 'r')
            data = {}
            for k,v in features_train[name_train.replace(".h5", "")].items():    
                if k == 'image_size':
                    data[k] = torch.from_numpy(v.__array__()).float()
                elif  k == 'descriptors':
                    data[k] = torch.from_numpy(v.__array__()[:,:self.num_feats]).float()
                else:
                    data[k] = torch.from_numpy(v.__array__()[:self.num_feats]).float()

            camera = self.infor.iloc[idx, 10:].to_numpy().astype(float)
            pose = self.infor.iloc[idx, 3:10].to_numpy().astype(float)
            
            # target
            name_target = self.infor.iloc[idx,2]
            h5File_target = osp.join(self.data_dir, "h5", name_target)
            features_target = h5py.File(h5File_target, 'r')
            target = {}
            target["camera"] = torch.from_numpy(camera).float()
            target["pose"] = torch.from_numpy(pose).float()
            tmp_pose = np.zeros(6)
            tmp_pose[:3] = pose[:3]
            tmp_pose[3:] = qlog(pose[3:])
            target["pose_nor"] = torch.from_numpy(tmp_pose).float()
            for k,v in features_target[name_target.replace(".h5", "")].items(): 
                if k =="p3D_ids":
                    v = v.__array__()
                    v[v!=-1] = 1
                    v[v==-1] = 0
                    target[k] = torch.from_numpy(v[:self.num_feats]).float()
                elif k == "p3Ds": 
                    target[k] = torch.from_numpy(v.__array__()[:self.num_feats].T).float() 
                else:
                    target[k] = torch.from_numpy(v.__array__()[:self.num_feats]).float()
            target["is_real_label"] = torch.from_numpy(np.array([1.])).float()
        else:
            idx = idx - self.length_train
            idx = self.list_unlabel_indexes[idx]
            # data
            name_train = self.unlabel_infor.iloc[idx,1]
            h5File_train = osp.join(self.unlabel_data_dir, "h5", name_train)
            features_train = h5py.File(h5File_train, 'r')
            data = {}
            for k,v in features_train[name_train.replace(".h5", "")].items():    
                data[k] = torch.from_numpy(v.__array__()).float()

            # target
            name_target = self.unlabel_infor.iloc[idx,2]
            h5File_target = osp.join(self.unlabel_data_dir, "h5", name_target)
            features_target = h5py.File(h5File_target, 'r')
            target = {}
            target["camera"] = torch.from_numpy(self.fake_camera).float()
            target["pose"] = torch.from_numpy(np.zeros(7)).float()
            target["pose_nor"] = torch.from_numpy(np.zeros(6)).float()
            target["errors"] = torch.from_numpy(np.zeros(2048)).float()
            target["xys"] = torch.from_numpy(np.zeros((2048,2))).float()
            for k,v in features_target[name_target.replace(".h5", "")].items(): 
                if k =="p3D_ids":
                    v = v.__array__()
                    v[v!=-1] = 1
                    v[v==-1] = 0
                    target[k] = torch.from_numpy(v).float()
                elif k == "p3Ds": 
                    target[k] = torch.from_numpy(v.__array__().T).float() 
                else:
                    target[k] = torch.from_numpy(v.__array__()).float()
            target["is_real_label"] = torch.from_numpy(np.array([0.])).float()
        return data, target

class _3DFeatLoc_Loader_Newest(Dataset):
    
    def __init__(self, data_dir, configs={"unlabel":False, "unlabel_rate":1.0, "augment":False}):
        self.data_dir = osp.join(data_dir, "train")
        self.infor = pd.read_csv(osp.join(self.data_dir, "readme.txt"), header=None, sep =" ")
        self.length_train = len(self.infor)
        self.num_feats = 2048
        self.configs = configs 
        self.length_unlabel = 0
        self.length_augment = 0


        if self.configs["augment"]:
            self.augment_data_dir = osp.join(data_dir, "augment")
            self.augment_infor = pd.read_csv(osp.join(self.augment_data_dir, "readme.txt"), header=None, sep=" ")
            self.length_augment = len(self.augment_infor)
            self.list_augment_indexes = list(range(self.length_augment))

        if self.configs["unlabel"]:
            self.unlabel_data_dir = osp.join(data_dir, "unlabel")
            self.unlabel_infor = pd.read_csv(osp.join(self.unlabel_data_dir, "readme.txt"), header=None, sep=" ")
            self.length_unlabel = len(self.unlabel_infor)
            self.list_unlabel_indexes = list(range(self.length_unlabel))
            random.shuffle(self.list_unlabel_indexes)
            self.list_unlabel_indexes = self.list_unlabel_indexes[:int(self.configs['unlabel_rate']*self.length_unlabel)]
            self.length_unlabel = len(self.list_unlabel_indexes)

        self.gen_train_idx() # init and shuffle indices. 
        print(self.length_train)
        print(self.length_augment)
        print(self.length_unlabel)
        print(self.length_idxs)
         

    def __len__(self):
        return self.length_idxs

    def gen_train_idx(self):
        '''
            This assumes length of augmented data same as 
        length of length of training data. 
            And length of unlabeled data is lower. 
        '''
        list_train_indexes = list(range(self.length_train))
        random.shuffle(list_train_indexes)
        list_augment_indexes = list(range(self.length_augment))
        random.shuffle(list_augment_indexes)
        list_unlabel_indexes = list(range(self.length_unlabel))
        random.shuffle(list_unlabel_indexes)

        # --- init 
        output = {}
        step = 2
        unlabel_step = 4
        augment_step = 1
        # --- end init
        start = 0
        stop = start + step
        count = 0
        start_unlabel = 0 
        stop_unlabel = start_unlabel + unlabel_step
        start_augment = 0
        stop_augment = start_augment + augment_step
        while start < self.length_train:
            tmp_train = list_train_indexes[start:stop]
            tmp_augment = list_augment_indexes[start_augment:stop_augment]
            tmp_unlabel = list_unlabel_indexes[start_unlabel:stop_unlabel]
            for idx_train in tmp_train:
                output[count] = [idx_train, "train"]
                count += 1
            for idx_unlabel in tmp_unlabel:
                output[count] = [idx_unlabel, "unlabel"]
                count += 1
            for idx_augment in tmp_augment:
                output[count] = [idx_augment, "augment"]
                count += 1
            start += step
            stop += step
            start_unlabel += unlabel_step 
            stop_unlabel += unlabel_step
            start_augment += augment_step
            stop_augment += augment_step
            if start_unlabel > self.length_unlabel:
                start_unlabel = 0 
                stop_unlabel = start_unlabel + unlabel_step
            if start_augment > self.length_augment:
                start_augment = 0 
                stop_augment = start_augment + augment_step

        self.train_idxs = output
        self.length_idxs = len(self.train_idxs)

    def get_infor(self, idx):
        real_idx = self.train_idxs[idx][0]
        mode = self.train_idxs[idx][1]
        if mode == "train":
            pandas_infor = self.infor
            data_dir = self.data_dir
        elif mode == "augment":
            pandas_infor = self.augment_infor
            data_dir = self.augment_data_dir
        elif mode == "unlabel":
            pandas_infor = self.unlabel_infor
            data_dir = self.unlabel_data_dir
        else:
            raise "Not implemented - mode -"
        return real_idx, pandas_infor, data_dir, mode

    
    def __getitem__(self, idx):
        real_idx, pd_infor, data_dir, mode = self.get_infor(idx)
        name_train = pd_infor.iloc[real_idx,1]
        h5File_train = osp.join(data_dir, "h5", name_train)
        features_train = h5py.File(h5File_train, 'r')
        data = {}
        for k,v in features_train[name_train.replace(".h5", "")].items():    
            if k == 'image_size':
                data[k] = torch.from_numpy(v.__array__()).float()
            elif  k == 'descriptors':
                data[k] = torch.from_numpy(v.__array__()).float()
            else:
                data[k] = torch.from_numpy(v.__array__()).float()


        camera = pd_infor.iloc[real_idx, 10:].to_numpy().astype(float)
        pose = pd_infor.iloc[real_idx, 3:10].to_numpy().astype(float)
        
        # target
        name_target = pd_infor.iloc[real_idx,2]
        h5File_target = osp.join(data_dir, "h5", name_target)
        features_target = h5py.File(h5File_target, 'r')
        target = {}
        target["camera"] = torch.from_numpy(camera).float()
        target["pose"] = torch.from_numpy(pose).float()
        tmp_pose = np.zeros(6)
        
        if mode != "train":
            target["pose_nor"] = torch.from_numpy(np.zeros(6)).float()
            target["errors"] = torch.from_numpy(np.zeros(2048)).float()
            target["xys"] = torch.from_numpy(np.zeros((2048,2))).float()
        else:
            tmp_pose[:3] = pose[:3]
            tmp_pose[3:] = qlog(pose[3:]) # normalize from 7dim to 6dim as APR MapNet
            target["pose_nor"] = torch.from_numpy(tmp_pose).float()

        for k,v in features_target[name_target.replace(".h5", "")].items(): 
            if k =="p3D_ids":
                v = v.__array__()
                v[v!=-1] = 1
                v[v==-1] = 0
                target[k] = torch.from_numpy(v).float()
            elif k == "p3Ds": 
                target[k] = torch.from_numpy(v.__array__().T).float() 
            else:
                target[k] = torch.from_numpy(v.__array__()).float()

        if mode == "train" or sum(target["camera"]) != 0:
            target["is_real_label"] = torch.from_numpy(np.array([1.])).float()
        else:
            target["is_real_label"] = torch.from_numpy(np.array([0.])).float()
            
        return data, target


class _3DFeatLoc_Loader_test(Dataset):
    
    def __init__(self, data_dir):
        self.data_dir = osp.join(data_dir, "test")
        self.infor = pd.read_csv(osp.join(self.data_dir, "readme.txt"), header=None, sep =" ")
        self.num_feats = 2048
        
    def __len__(self):
        return len(self.infor)
    
    def __getitem__(self, idx):
        # input 
        name_train = self.infor.iloc[idx,1]
        h5File_train = osp.join(self.data_dir, "h5", name_train)
        features_train = h5py.File(h5File_train, 'r')
        data = {}
        data['image_name'] = self.infor.iloc[idx,0]
        for k,v in features_train[name_train.replace(".h5", "")].items():    
            if k == 'image_size':
                data[k] = torch.from_numpy(v.__array__()).float()
            elif  k == 'descriptors':
                data[k] = torch.from_numpy(v.__array__()[:,:self.num_feats]).float()
            else:
                data[k] = torch.from_numpy(v.__array__()[:self.num_feats]).float()
            # data[k] = torch.from_numpy(v.__array__()).float()
        camera = self.infor.iloc[idx, 9:].to_numpy().astype(float)
        data["camera"] = torch.from_numpy(camera).float()
        return data


################################################## Thuan test with changing number features 

class _3DFeatLoc_Loader_test_change_num(Dataset):
    
    def __init__(self, data_dir, num_feats):
        self.data_dir = osp.join(data_dir, "test")
        self.infor = pd.read_csv(osp.join(self.data_dir, "readme.txt"), header=None, sep =" ")
        self.num_feats = num_feats
        
    def __len__(self):
        return len(self.infor)
    
    def __getitem__(self, idx):
        # input 
        name_train = self.infor.iloc[idx,1]
        h5File_train = osp.join(self.data_dir, "h5", name_train)
        features_train = h5py.File(h5File_train, 'r')
        data = {}
        for k,v in features_train[name_train.replace(".h5", "")].items():    
            if k == 'image_size':
                data[k] = torch.from_numpy(v.__array__()).float()
            elif  k == 'descriptors':
                data[k] = torch.from_numpy(v.__array__()[:,:self.num_feats]).float()
            else:
                data[k] = torch.from_numpy(v.__array__()[:self.num_feats]).float()
            # data[k] = torch.from_numpy(v.__array__()).float()
        camera = self.infor.iloc[idx, 9:].to_numpy().astype(float)
        data["camera"] = torch.from_numpy(camera).float()
        return data



if __name__ == "__main__":
    scene = "scene6"
    data_dir  = "../dataset/indoor6"
    import os 
    '''
    dataset = _3DFeatLoc_Loader(osp.join(data_dir, scene))
    print("TEST _3DFeatLoc_Loader")
    print(len(dataset))
    print(dataset[0])
    '''
    '''
    dataset = _3DFeatLoc_Loader_test(osp.join(data_dir, scene))
    print("TEST _3DFeatLoc_Loader_test")
    print(len(dataset))
    print(dataset[0])
    '''
    dataset_dir = "/home/thuan/Desktop/GITHUB/FeatLoc_preparation/dataset/Hierarchical_Localization/datasets/"
    configs={"unlabel":True, "unlabel_rate": 1.0, "augment":True, "imgdata_dir":os.path.join(dataset_dir, "indoor6", "indoor6_sfm_triangulated" , scene)}
    dataset = _3DFeatLoc_Loader_New(osp.join(data_dir, scene), configs = configs)
    print(len(dataset))
    for i in range(len(dataset)):
        # print(i)
        dataset[i]
        if i > 100:
            break
    print("--done--")
