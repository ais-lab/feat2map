#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 13:29:13 2022

@author: thuan.aislab@gmail.com
"""

import torch
import sys
import os.path as osp
import os
sys.path.append(osp.join(osp.dirname(__file__), ".."))
from utils.read_write_model import read_model, read_images_text, read_images_binary
import utils.utils as uuls 
import h5py 
from tqdm import tqdm
import numpy as np
import pandas as pd

import copy

# from third_party.matching import Matching
from third_party.matching_thuan import Matching
from PIL import Image
import PIL
import cv2
import torchvision.transforms.functional as F
import torchvision.transforms as T
import random



def preprocessing(dataset:str, hloc_out_dir:str, dataset_dir:str, \
                  use_depth:bool, scene:str, out_dir:str):
    """
    This function will generate train and test data of 3D feature positions 
    based on hloc toolbox. 
    
    Parameters
    ----------
    dataset : str
        Name of dataset ex: 7Scenes
    hloc_out_dir : str
        The output sfm directory of hloc toolbox.
    dataset_dir : str
        The directory of the dataset in hloc toolbox.
    use_depth : bool
        Wheather use 3D points clouds that have been filtered with depth.
    scene : str
        scene name.
    out_dir : str
        Output path for the data.
    Returns
    -------
    None.

    """
    print("--------  Generating Training and Testing data ---------  \n")

    if dataset == "7scenes":
        sift_sfm_dir = osp.join(dataset_dir, dataset, "7scenes_sfm_triangulated" ,\
                                scene, "triangulated") # only used for extracting test labels
        # get test list
        testlist_dir = osp.join(sift_sfm_dir, "list_test.txt")
        sift_images = read_images_text(osp.join(sift_sfm_dir, "images.txt"))
        sift_cameras_class = uuls.Cambridge_Cameras(osp.join(hloc_out_dir, dataset, scene, "query_list_with_intrinsics.txt"))
        vallist_dir = None
    elif dataset == "Cambridge":
        sift_sfm_dir = osp.join(dataset_dir, dataset, "CambridgeLandmarks_Colmap_Retriangulated_1024px" ,\
                                scene)
        # get test list
        testlist_dir = osp.join(sift_sfm_dir, "list_query.txt")
        sift_sfm_dir = osp.join(sift_sfm_dir, "empty_all")
        sift_images = read_images_text(osp.join(sift_sfm_dir, "images.txt"))
        sift_cameras_class = uuls.Cambridge_Cameras(osp.join(hloc_out_dir, dataset, scene, "query_list_with_intrinsics.txt"))
        vallist_dir = None
    elif dataset =="12scenes":
        sift_sfm_dir = osp.join(dataset_dir, dataset, "12scenes_sfm_triangulated" ,scene) # only used for extracting test labels
        # get test list
        testlist_dir = osp.join(sift_sfm_dir, "list_test.txt")
        sift_images = read_images_binary(osp.join(sift_sfm_dir, "images.bin"))
        sift_cameras_class = uuls.Cambridge_Cameras(osp.join(hloc_out_dir, dataset, scene, "query_list_with_intrinsics.txt"))
        vallist_dir = None
    elif dataset =="indoor6":
        sift_sfm_dir = osp.join(dataset_dir, dataset, "indoor6_sfm_triangulated" ,scene) # only used for extracting test labels
        # get test list
        testlist_dir = osp.join(sift_sfm_dir, "list_test.txt")
        vallist_dir = osp.join(sift_sfm_dir, "list_test_val.txt")
        sift_images = read_images_binary(osp.join(sift_sfm_dir, "images.bin"))
        sift_cameras_class = uuls.Cambridge_Cameras(osp.join(hloc_out_dir, dataset, scene, "query_list_with_intrinsics.txt"))
    elif dataset =="BKC":
        sift_sfm_dir = osp.join(dataset_dir, dataset ,scene) # only used for extracting test labels
        # get test list
        testlist_dir = osp.join(sift_sfm_dir, "test_seq4.txt")
        vallist_dir = None
        sift_images = read_images_binary(osp.join(sift_sfm_dir, "images.bin"))
        sift_cameras_class = uuls.Cambridge_Cameras(osp.join(hloc_out_dir, dataset, scene, "seq4_query_list_with_intrinsics.txt"))
    else:
        raise "Not implmented"
    
    with open(testlist_dir,'r') as f:
        listnames_test = f.read().rstrip().split('\n')
    if vallist_dir is not None:
        with open(vallist_dir,'r') as f:
            listnames_test_val = f.read().rstrip().split('\n')
    else:
        listnames_test_val = []
    
    if use_depth:
        sfm_path_train = osp.join(hloc_out_dir, dataset, scene, "sfm_superpoint+superglue+depth")
    else:
        sfm_path_train = osp.join(hloc_out_dir, dataset, scene, "sfm_superpoint+superglue")
        
    out_dir = uuls.makedir_OutScene(out_dir, dataset, scene)
    if out_dir is None:
        print("-------------- DONE -------------- \n")
        return 0
    
    features = h5py.File(osp.join(hloc_out_dir, dataset, scene, "feats-superpoint-n4096-r1024.h5"), 'r')

    cameras_train, images_train, points3D_train = read_model(sfm_path_train)
    ###### -------------- save mean of 3d points 
    mean_coords = []
    for pid, pointclass in points3D_train.items():
        mean_coords.append(pointclass.xyz)
    mean_coords = np.array(mean_coords)
    mean_coords = np.mean(mean_coords, 0)
    np.savetxt(osp.join(out_dir, "mean.txt"), mean_coords, delimiter=" ")
    ##### -------------- end save 3d meaned points 
    train_name2id = {image.name: i for i, image in images_train.items()}
    sift_name2id = {image.name: i for i, image in sift_images.items()}
    
    i = 0
    j = 0 
    for id_, image in tqdm(sift_images.items()):
        t_name = image.name
        if t_name in listnames_test:
            # test data.
            mode = "test"
            s_name = "test_" + str(i) + ".h5"
            sfm_id = sift_name2id[t_name]
            pose = uuls.text_pose(sift_images[sfm_id].tvec, sift_images[sfm_id].qvec)
            camera = sift_cameras_class.get_camera(t_name)
            i += 1
            with open(osp.join(out_dir, mode, "readme.txt"), "a") as wt:
                wt.write("{0} {1} {2} {3}\n".format(*[t_name, s_name, pose, camera]))
        elif t_name in listnames_test_val:
            # val data.
            continue
        else:
            try:
                # train data.
                mode = "train"
                s_name = "train_" + str(j) + ".h5"
                s_name3d = "label_" + str(j) + ".h5"
                sfm_id = train_name2id[t_name]
                pose = uuls.text_pose(images_train[sfm_id].tvec, images_train[sfm_id].qvec)
                
                if dataset == "7scenes" or "12scenes":
                    camera = uuls.camera2txt(cameras_train[1])
                else:
                    camera = uuls.camera2txt(cameras_train[sfm_id])
                
                p3D_ids = images_train[sfm_id].point3D_ids
                xys = images_train[sfm_id].xys
                
                if not p3D_ids.size > 0:
                    continue
                
                p3Ds = np.stack([points3D_train[ii].xyz if ii != -1 else 
                                 np.array([0,0,0]) for ii in p3D_ids], 0)
                errors = np.stack([points3D_train[ii].error if ii != -1 else 
                                 np.array(0) for ii in p3D_ids], 0)

                assert len(p3D_ids) == len(xys) == len(p3Ds) == len(errors)
                
                data_3D = {}
                data_3D = {"p3D_ids":p3D_ids, "xys": xys, "p3Ds": p3Ds, "errors":errors}
                with h5py.File(osp.join(out_dir, mode, "h5", s_name3d), "w") as fd: 
                    grp = fd.create_group(s_name3d.replace(".h5", ""))
                    for k, v in data_3D.items():
                        grp.create_dataset(k, data=v)
                j += 1
                with open(osp.join(out_dir, mode, "readme.txt"), "a") as wt:
                    wt.write("{0} {1} {2} {3} {4}\n".format(*[t_name, s_name, s_name3d, pose, camera]))
            except:
                continue

        data= {}
        for k,v in features[t_name].items():   
            data[k] = v.__array__()

        with h5py.File(osp.join(out_dir, mode, "h5", s_name), "w") as fd: 
            grp = fd.create_group(s_name.replace(".h5", ""))
            for k, v in data.items():
                grp.create_dataset(k, data=v)

    print("-------------- DONE -------------- \n")


def top2dicts(path):
    data = pd.read_csv(path, sep =" ", header=None)
    out_data = {}
    for i in range(len(data)):
        temp = data.iloc[i,0]
        if temp not in out_data: 
            out_data[temp] = [data.iloc[i,1]]
        else:
            out_data[temp].append(data.iloc[i,1])
    return out_data

def train2dicts(path):
    data =pd.read_csv(path, sep = " ", header = None)
    out_data = {}
    out_data = {data.iloc[i,0]: [data.iloc[i,0]] for i in range(len(data))}
    return out_data

def gen_dict2trainInfor(path):
    # path: the path to train data folder
    data = pd.read_csv(osp.join(path, "readme.txt"), sep = " ", header = None)
    out_pose_camera_list = []
    out_data = {}
    for i in range(len(data)):
        out_data[data.iloc[i,0]] = (data.iloc[i,1], data.iloc[i,2])
        out_pose_camera_list.append(data.iloc[i,3:])
    return out_data, out_pose_camera_list


def read_image(ref_path, grayscale=False, aumgent = False, unlabel = False, only_brightness=False):
    img = Image.open(ref_path)
    if img is None:
        raise ValueError('Cannot read image {}'.format(ref_path))
    if grayscale:
        img = img.convert('L')
        if aumgent:
            if unlabel:
                br_factor = random.uniform(0.4, 0.8)
                img = F.adjust_brightness(img, br_factor)
                # if not only_brightness:
                #     perspective_transformer = T.RandomPerspective(distortion_scale=0.4, p=1.0)
                #     img = perspective_transformer(img)
                # else:
                #     tmp_m,tmp_n = img.size
                #     crop_and_resize_transform = T.RandomResizedCrop(size=(tmp_n, tmp_m), 
                #                                              scale=(0.7, 0.8))
                #     img = crop_and_resize_transform(img)
            else:
                br_factor1 = random.uniform(0.5, 0.85)
                br_factor2 = random.uniform(1.15, 1.5)
                br_factor = random.choice([br_factor1, br_factor2])
                img = F.adjust_brightness(img, br_factor)
                # jiter_transform = T.ColorJitter(brightness=(0.7,1.2),contrast=(0,0.5))
                # img = jiter_transform(img)
                if not only_brightness:
                    # if random.choice([True, False]):
                    perspective_transformer = T.RandomPerspective(distortion_scale=0.4, p=1.0)
                    img = perspective_transformer(img)
                    # else:
                    #     tmp_m,tmp_n = img.size
                    #     crop_and_resize_transform = T.RandomResizedCrop(size=(tmp_n, tmp_m), 
                    #                                              scale=(0.6, 0.8))
                    #     img = crop_and_resize_transform(img)


        return img 
    else:
        return img

def resize_image(image, size, interp):
    if interp.startswith('cv2_'):
        # for opencv
        interp = getattr(cv2, 'INTER_'+interp[len('cv2_'):].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    elif interp.startswith('pil_'):
        # for PIL
        interp = getattr(PIL.Image, interp[len('pil_'):].upper())
        resized = image
        resized = resized.resize(size, resample=interp)
    else:
        raise ValueError(
            f'Unknown interpolation {interp}.')
    return resized




def map_tensor(input_, func):
    if isinstance(input_, torch.Tensor):
        return func(input_)
    elif isinstance(input_, string_classes):
        return input_
    elif isinstance(input_, collections.Mapping):
        return {k: map_tensor(sample, func) for k, sample in input_.items()}
    elif isinstance(input_, collections.Sequence):
        return [map_tensor(sample, func) for sample in input_]
    else:
        raise TypeError(
            f'input must be tensor, dict or list; found {type(input_)}')





def processing_unlabeldata(dataset:str, dataset_dir:str, scene:str, out_dir:str, configs:dict,
    aumgent = False, unlabel = False, augment_unlabel=False):
    print("--------  Generating Pseudo data from unlabeled ones ---------")

    mode = "unlabel"
    Extractor = list(configs['extractor'])[0]
    if dataset == "7scenes":
        data_dir = osp.join(dataset_dir, dataset, scene)
        train_dir = osp.join(out_dir, dataset, scene, "train")

    elif dataset == "Cambridge":
        # this will be corrected first 
        data_dir = osp.join(dataset_dir, dataset, scene)
        top_matches_dir = osp.join(out_dir, dataset, scene + "-netvlad10.txt")
        train_dir = osp.join(out_dir, dataset, scene, "train")
        if Extractor == 'superpoint':
        	configs['extractor']['superglue']['weights'] = 'outdoor'
    elif dataset =="12scenes":
        data_dir = osp.join(dataset_dir, dataset, scene)
    elif dataset == "indoor6":
        data_dir = osp.join(dataset_dir, dataset, "indoor6_sfm_triangulated" ,scene)
        configs['extractor']['superglue']['weights'] = 'indoor'
        configs['process_image']['resize_max'] = 640
        train_dir = osp.join(out_dir, dataset, scene, "train")
        top_matches_dir = osp.join(out_dir, dataset, scene + "-val-netvlad10.txt")
        top_matches_dir_test = osp.join(out_dir, dataset, scene + "-test-netvlad10.txt")
    elif dataset == "BKC":
        data_dir = osp.join(dataset_dir, dataset ,scene)
        configs['extractor']['superglue']['weights'] = 'outdoor'
        configs['process_image']['resize_max'] = 1024
        train_dir = osp.join(out_dir, dataset, scene, "train")
        top_matches_dir = osp.join(out_dir, dataset, scene + "-netvlad10.txt")
        # top_matches_dir_test = osp.join(out_dir, dataset, scene + "-test-netvlad10.txt")

    else:
        raise "Not implmented"

    

    name2infors, pose_camera_list = gen_dict2trainInfor(train_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    if Extractor == "sift":
    	device ='cpu' 
    matching = Matching(configs).eval().to(device)

    def run_single_data(configs, tmp_match_dict, path_out, iii, aumgent, start_id, unlabel):
        
        unlabeled_name = list(tmp_match_dict.keys())[iii]
        matches_list = tmp_match_dict[unlabeled_name]
        
        if aumgent and (unlabel is False) and (random.random() > 0.4):
            only_brightness = True
        else:
            only_brightness = False
        
        def do_matching(configs, tmp_match_dict, path_out, iii, aumgent, start_id, unlabel, vlad_i, unlabeled_name, matches_list):
            labeled_name = matches_list[vlad_i]
            name_train = name2infors[labeled_name][0]
            name_target = name2infors[labeled_name][1]
            h5File_train = osp.join(train_dir, "h5", name_train)
            h5File_target = osp.join(train_dir, "h5", name_target)
            # Load sparse features.
            features_train = h5py.File(h5File_train, 'r')
            labeled_data = {}
            for k2, d2 in features_train[name_train.replace(".h5", "")].items():    
                labeled_data[k2] = torch.from_numpy(d2.__array__()).float()
            # Load labeled data.
            features_target = h5py.File(h5File_target, 'r')
            labeled_target = {}
            for k,v in features_target[name_target.replace(".h5", "")].items(): 
                labeled_target[k] = v.__array__()
     

            unlabeled_img_pill = read_image(osp.join(data_dir, unlabeled_name), configs['process_image']['grayscale'], aumgent, unlabel, only_brightness)

            labeled_image_pill = read_image(osp.join(data_dir, labeled_name), configs['process_image']['grayscale'])
    
            size = unlabeled_img_pill.size
    
            if max(size) > configs['process_image']['resize_max']:
                scale = configs['process_image']['resize_max'] / max(size)
                size_new = tuple(int(round(x*scale)) for x in size)
                unlabeled_img = resize_image(unlabeled_img_pill, size_new, configs['process_image']['interpolation'])
            else:
                scale = 1.0
                unlabeled_img = copy.deepcopy(unlabeled_img_pill)
                size_new = size

            # for drawing
            unlabeled_img_draw = copy.deepcopy(unlabeled_img)
            labeled_image_pill_draw = resize_image(labeled_image_pill, size_new, configs['process_image']['interpolation'])
            
            ## convert data to suitable torch tensor with cuda. 
            unlabeled_img = np.asarray(unlabeled_img)
            labeled_image = np.asarray(labeled_image_pill)
            if configs['process_image']['grayscale']:
                unlabeled_img = unlabeled_img[None]
                labeled_image = labeled_image[None]
            else:
                unlabeled_img = unlabeled_img.transpose((2, 0, 1))  # HxWxC to CxHxW
                labeled_image = labeled_image.transpose((2, 0, 1))  # HxWxC to CxHxW
            unlabeled_img = unlabeled_img / 255.
            labeled_image = labeled_image / 255.
    
            # convert to tensor.
            unlabeled_img = torch.from_numpy(unlabeled_img).float()
            unlabeled_img = torch.unsqueeze(unlabeled_img, dim = 0) 
            labeled_image = torch.from_numpy(labeled_image).float()
            labeled_image = torch.unsqueeze(labeled_image, dim = 0) 
    
            matching_data ={"image0": unlabeled_img, "image1": labeled_image,  "keypoints1": torch.unsqueeze(labeled_data['keypoints'], dim = 0),
            "descriptors1": torch.unsqueeze(labeled_data['descriptors'], dim = 0), "scores1": torch.unsqueeze(labeled_data['scores'], dim = 0)}
    
            for k3, d3 in matching_data.items():    
                matching_data[k3] = map_tensor(d3, lambda x: x.to(device))
    
            pred_matches = matching(matching_data) # matching which use SuperGlue. 
            
            save_data = {}
            save_data["keypoints"] = pred_matches['keypoints0'][0].detach().cpu().numpy()

            if max(size) > configs['process_image']['resize_max']:
                save_data["keypoints"] = (save_data["keypoints"] + .5) / scale - .5

            save_data["descriptors"] = pred_matches['descriptors0'][0].detach().cpu().numpy()
            save_data["scores"] = pred_matches['scores0'][0].detach().cpu().numpy()
            save_data["image_size"] = np.asarray(size_new)
    
            pred_matchesnp = torch.squeeze(pred_matches["matches0"]).detach().cpu().numpy()

    
            temp_unmatches = pred_matchesnp == -1
            save_target = {"p3D_ids":labeled_target['p3D_ids'][pred_matchesnp], "p3Ds": labeled_target['p3Ds'][pred_matchesnp]} # re-sort
            save_target["p3D_ids"][temp_unmatches] = -1  # further update if not matched
            save_target["p3Ds"][temp_unmatches] = np.array([0.,0.,0.]) # further update if not matched
    
    
    
            # --------------- Visualization --------------------------------
            if False: # True -> to visualize the results
                import matplotlib.cm as cm
                kpts0 = pred_matches['keypoints0'][0].detach().cpu().numpy()
                kpts1 = pred_matches['keypoints1'][0].detach().cpu().numpy()*scale
                matches = pred_matches['matches0'][0].detach().cpu().numpy()
        
                confidence = pred_matches['matching_scores0'][0].detach().cpu().numpy()
        
                valid = matches > -1
                mkpts0 = kpts0[valid]
                mkpts1 = kpts1[matches[valid]]
                color = cm.jet(confidence[valid])
                text = [
                    'SuperGlue',
                    'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                    'Matches: {}'.format(len(mkpts0))
                ]
                out = uuls.make_matching_plot_fast(
                        np.asarray(unlabeled_img_draw), np.asarray(labeled_image_pill_draw), kpts0, kpts1, mkpts0, mkpts1, color, text,
                        path = '/home/pc1/Desktop/Dash/' + str(iii) + '_' + str(vlad_i) +'.png')
                print("Saved visualized matched images")

                if iii > 10: # 
                    raise
            
            #---------------- END VISUALIZATION

            return save_data, save_target
        
        
        def merge_target_list(target_list):
            # get index
            length_targetlist = len(target_list)
            n_succ_matches_list = [len(target_list[i]["p3D_ids"][target_list[i]["p3D_ids"] != -1]) for i in range(length_targetlist)]
            max_val_idx = n_succ_matches_list.index(max(n_succ_matches_list))
            
            
            ref_target = target_list[max_val_idx]
            for i in range(length_targetlist):
                empty_indexes = np.where(ref_target["p3D_ids"] == -1)
                if n_succ_matches_list[i] > 50 and i != max_val_idx:
                    ref_target["p3D_ids"][empty_indexes] = target_list[i]["p3D_ids"][empty_indexes]
                    ref_target["p3Ds"][empty_indexes] = target_list[i]["p3Ds"][empty_indexes]
                else:
                    continue
                
            return ref_target, len(ref_target["p3D_ids"][ref_target["p3D_ids"] != -1])
        
        
        save_target_list = []
        for vlad_i in range(10):
            save_data, tmp_save_target = do_matching(configs, tmp_match_dict, path_out, 
                iii, aumgent, start_id, unlabel, vlad_i, unlabeled_name, matches_list)
            save_target_list.append(tmp_save_target)
            if not unlabel: 
                break
        
        if unlabel:
            save_target, num_success_matched = merge_target_list(save_target_list)
        else:
            save_target = tmp_save_target
            num_success_matched = len(save_target["p3D_ids"][save_target["p3D_ids"] != -1])
           
        if aumgent and (unlabel is False) and only_brightness:
            pose = pose_camera_list[iii][:7].to_numpy()
            camera = pose_camera_list[iii][7:].to_numpy()
        else:
            pose = np.zeros(7)
            camera = np.zeros(5) if dataset == "7scenes" else np.zeros(6) 
        pose = uuls.numpy2str(pose)
        camera = uuls.numpy2str(camera)

        s_name = "unldata_" + str(int(iii+start_id)) + ".h5"
        s_name3d = "plabel_" + str(int(iii+start_id)) + ".h5"

        save_target["xys"] = save_data["keypoints"]

        for k in save_data:
            save_data[k] = save_data[k].astype(np.float16)
        
        with h5py.File(osp.join(path_out, "h5", s_name3d), "w") as fd: 
            grp = fd.create_group(s_name3d.replace(".h5", ""))
            for k, v in save_target.items():
                grp.create_dataset(k, data=v)

        with open(osp.join(path_out, "readme.txt"), "a") as wt:
            wt.write("{0} {1} {2} {3} {4}\n".format(*[unlabeled_name, s_name, s_name3d, pose, camera]))

        with h5py.File(osp.join(path_out, "h5", s_name), "w") as fd: 
            grp = fd.create_group(s_name.replace(".h5", ""))
            for k, v in save_data.items():
                grp.create_dataset(k, data=v)       
        torch.cuda.empty_cache()
        
        return num_success_matched

    start_id = 0
    if aumgent:
        print("--------------Augmenting-------------------- \n")

        # for augmentation the training data. 
        path_out = osp.join(out_dir, dataset, scene, "augment")
        aug_match_dict = train2dicts(osp.join(train_dir, "readme.txt")) 
        length_aug_match_dict = len(aug_match_dict)
        num_success_matched_list = []
        for xxx in tqdm(range(length_aug_match_dict)):
            # try:
            tmpnum_success_matched = run_single_data(configs, aug_match_dict, path_out, xxx, True, 0, False)
            num_success_matched_list.append(tmpnum_success_matched)
            # except:
            #     continue
        print("Average num of all matched each image: ", np.mean(num_success_matched_list))
        # print(num_success_matched_list)
        start_id = length_aug_match_dict
    if unlabel:
        # apply with pure unlabel data. 
        print("--------------Processing on unlabeled data--------------------  \n")
        # path_out = osp.join(out_dir, dataset, scene, "unlabel")
        path_out = osp.join(out_dir, dataset, scene, "augment")
        unlabel_match_dict = top2dicts(top_matches_dir)
        num_success_matched_list = []
        for xxx in tqdm(range(len(unlabel_match_dict))):
            tmpnum_success_matched = run_single_data(configs, unlabel_match_dict, path_out, xxx, False, start_id, unlabel)
            num_success_matched_list.append(tmpnum_success_matched)
        
        print("Average num of all matched each image: ", np.mean(num_success_matched_list))

        if augment_unlabel:
            print("--------------Augmenting on purely unlabeled data--------------------  \n")
            # apply augmentation with unlabel data. 
            start_id = start_id + len(unlabel_match_dict)
            num_success_matched_list = []
            for xxx in tqdm(range(len(unlabel_match_dict))):
                tmpnum_success_matched = run_single_data(configs, unlabel_match_dict, path_out, xxx, True, start_id, unlabel)
                num_success_matched_list.append(tmpnum_success_matched)
        
            print("Average num of all matched each image: ", np.mean(num_success_matched_list))


    print("-------------- DONE -------------- ")




if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    dataset = "Cambridge"
    scene = "KingsCollege"
    process_train_data_augmentation = True   # do augmentation for training data.
    process_unlabel_data = False # generate pseudo data from unlabels.
    process_unlabel_data_pls_augment = False # do augmentation on unlabel data.

    configs = {
            'extractor':{
                'superpoint': {
                    'nms_radius': 3,
                    'keypoint_threshold': 0.0,
                    'max_keypoints': 2048
                },
                'superglue': {
                    'weights': 'indoor',
                    'sinkhorn_iterations': 70,
                    'match_threshold': 0.2,
                },
            },
            "process_image":
                {
                'grayscale': True,
                'resize_max': 1024, # this for cambridge dataset.
                'interpolation': 'pil_linear', 
                }
        }
    # configs = {
    #         'extractor':{
    #             'sift': {
    #                 'max_keypoints': 2048
    #             },
    #         },
    #         "process_image":
    #             {
    #             'grayscale': True,
    #             'resize_max': 1600, 
    #             'interpolation': 'pil_linear', 
    #             }
    #     }
    # ---------------
    hloc_out_dir = "../third_party/Hierarchical_Localization/outputs/"
    dataset_dir = "../third_party/Hierarchical_Localization/datasets/"
    out_dir = "../dataset/"
    
    if dataset == "7scenes" or dataset == "12scenes":
        use_depth = True
    else:
        use_depth = False
    use_depth = False

    # End initializing the parameters

    print("Working on: ", scene, " scene")
    preprocessing(dataset, hloc_out_dir, dataset_dir, use_depth, scene, out_dir)
    processing_unlabeldata(dataset, dataset_dir, scene, out_dir, configs, aumgent = process_train_data_augmentation, unlabel = process_unlabel_data, 
        augment_unlabel = process_unlabel_data_pls_augment)
