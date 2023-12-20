#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 19:02:43 2022

@author: thuan
"""

from processing.dataloader import _3DFeatLoc_Loader, _3DFeatLoc_Loader_test, _3DFeatLoc_Loader_New
import argparse 
import os
from models import criterion
from utils.select_model import select_model 
from models.evaluator import Evaluator


parser = argparse.ArgumentParser(description='Training script for 3DFeatLoc and'
                                 'its variants')

parser.add_argument('--dataset',type=str,default="Cambridge",
                    help="name of the dataset")
parser.add_argument('--scene',type=str,default="KingsCollege",
                    help="name of scene")
parser.add_argument('--config_file',type=str,default="configs/configsV0.ini",
                    help="configs path")
parser.add_argument('--model',type=int,default=2,
                    help="choose the model to be trained")
parser.add_argument('--single', type=bool, 
                    default=False, help = 'if True, evaluate only a single trained model')
parser.add_argument('--eval_train', type=bool, 
                    default=False)
parser.add_argument('--epoch', type=int, 
                    default=990)
parser.add_argument('--cudaid', type=int, 
                    default=1)
parser.add_argument('--uncer', type=bool, 
                    default=True)
parser.add_argument('--uncer_thres', type=float, 
                    default=0.5)
parser.add_argument('--start_epoch', type=int, 
                    default=0)
parser.add_argument('--stop_epoch', type=int, 
                    default=0)
parser.add_argument('--unlabel', type=bool, default=0,
                    help='was model trained with unlabel?')
parser.add_argument('--unlabel_rate', type=float, default=1.0,
                    help='percentage of unlabel data used to train')
parser.add_argument('--augment', type=int, default=0, choices =[0,1],
                    help='apply data augementation or not')
parser.add_argument('--idx_eval', type=int, default=0,
                    help='threshold of evaluation 0 (5cm),1 (15cm), etc,...')



class Find_Best(object):
    def __init__(self, idx_eval):
        # meter:    0.05, 0.15 ,0.22, 0.35, 0.38 0.5
        # idx_eval:   0     1     2     3     4   5
        self.idx_eval = idx_eval
        self.epoches = []
        self.mean_ts = []
        self.uncer_thres_list_for_t = []
        self.mean_rs = []
        self.ratios = []
    def update(self, epoch, med_t, med_R, ratio, uncer_thres=0):
        self.epoches.append(epoch)
        self.mean_ts.append(med_t)
        self.uncer_thres_list_for_t.append(uncer_thres)
        self.mean_rs.append(med_R)
        self.ratios.append(ratio[self.idx_eval])
    def final_infor(self):
        best_meant = min(self.mean_ts)
        idx_t = self.mean_ts.index(best_meant)
        best_meanR = min(self.mean_rs)
        idx_R = self.mean_rs.index(best_meanR)
        best_ratio = max(self.ratios)
        idx_r = self.ratios.index(best_ratio)
        print("Best meand t :", best_meant, "epoches: ", self.epoches[idx_t], "uncer_thres: ", self.uncer_thres_list_for_t[idx_t])
        print("Best meand R :", self.mean_rs[idx_R], "epoches: ", self.epoches[idx_R])
        print("Best ratio :", best_ratio, "epoches: ", self.epoches[idx_r])
        return self.epoches[idx_t], self.epoches[idx_R], self.uncer_thres_list_for_t[idx_t], self.epoches[idx_r], self.uncer_thres_list_for_t[idx_r]
    def save(self, path):
        import pandas as pd 
        import numpy as np 
        tmp = np.zeros((len(self.epoches), 5))
        tmp = pd.DataFrame(tmp)
        tmp.iloc[:,0] = self.epoches
        tmp.iloc[:,1] = self.mean_ts
        tmp.iloc[:,2] = self.mean_rs
        tmp.iloc[:,3] = self.ratios
        tmp.iloc[:,4] = self.uncer_thres_list_for_t
        tmp.to_csv(path, header=False, sep=" ", index=False)

args = parser.parse_args()
best  = Find_Best(args.idx_eval)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cudaid)
config_name = args.config_file.replace("configs/", "")
config_name = config_name.split('.')[0]



model, model_name = select_model(args.model)



# criterion 
train_criterion = criterion._3DFLCriterion()
test_criterion = None

experiment_name = '{:s}_{:s}_{:s}_{:s}'.format(args.dataset, args.scene, config_name, model_name)
global_eval_path = 'logs/' + experiment_name

# dataset
data_path = os.path.join("dataset/" + args.dataset, args.scene)

####### --- only for calculating original num_epoch

# data loader configs
dataset_dir = "../third_party/Hierarchical_Localization/datasets/"
import os.path as osp
if args.dataset == "7scenes":
    data_dir = osp.join(dataset_dir, args.dataset, args.scene)
elif args.dataset == "Cambridge":
    # this will be corrected first 
    data_dir = osp.join(dataset_dir, args.dataset, args.scene)

elif args.dataset =="12scenes":
    data_dir = osp.join(dataset_dir, args.dataset, args.scene)
elif args.dataset == "indoor6":
    data_dir = osp.join(dataset_dir, args.dataset, "indoor6_sfm_triangulated" , args.scene)
elif args.dataset == "BKC":
    data_dir = osp.join(dataset_dir, args.dataset , args.scene)
else:
    raise "Not implmented"

data_loader_configs = {
    "unlabel": args.unlabel, 
    "unlabel_rate": args.unlabel_rate,
    "augment": args.augment,
    "imgdata_dir": data_dir,
}
train_set_for_cal_nepoch = _3DFeatLoc_Loader_New(os.path.join("dataset/" + args.dataset, args.scene), configs = data_loader_configs)

###### --- end only 



train_set = _3DFeatLoc_Loader(data_path)
test_set = _3DFeatLoc_Loader_test(data_path)

import configparser
settings = configparser.ConfigParser()
with open(args.config_file, 'r') as f:
  settings.read_file(f)
config = {}
section = settings['training']
batch_size = section.getint('batch_size')
config['n_epochs'] = batch_size * int(section.getint('n_iters')/len(train_set_for_cal_nepoch))
config['n_epochs_add'] = batch_size * int(section.getint('n_iters_add')/len(train_set_for_cal_nepoch))


if args.start_epoch == 0:
    saved_epochs = [int((config['n_epochs']+config['n_epochs_add'])*0.9+1),  int(config['n_epochs']+config['n_epochs_add'])]
    folder_name = "val"
else:
    saved_epochs = [args.start_epoch , args.stop_epoch]
    folder_name = "val_2"

print("Evaluating from epoch {} to {}".format(*saved_epochs))


for thres_unc in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
    if args.single:
        thres_unc = args.uncer_thres
    uncertainty_config = {'state':args.uncer, 'threshold':thres_unc}
    if uncertainty_config['state']:
        print('testing with UNCERTAINTY, threshold = {}'.format(uncertainty_config['threshold']))
    for epoch in range(saved_epochs[0], saved_epochs[1]):
        try:
            if args.single:
                epoch = args.epoch
                eval_train = args.eval_train
            else:
                eval_train = False
            
            print("Working on scene: {}, is_uncer: {}, uncer_thres: {}".format(args.scene, args.uncer, thres_unc))
            
            checkpoint =  global_eval_path + '/epoch_{:03d}.pth.tar'.format(epoch)
            
            print(checkpoint)
            
            evaler = Evaluator(experiment_name, model, test_criterion, checkpoint, test_set,
                                train_set, data_path, train_criterion, args.dataset, eval_train = eval_train, 
                                w_uncertainty=uncertainty_config, model_ver=args.model, folder_name = folder_name)
            mean_t, med_R, ratio = evaler.evaler()
            best.update(epoch, mean_t, med_R, ratio, thres_unc)
            if args.single:
                raise
        except:
            if args.single:
                raise
            print("ERROR AT UNCERTAINTY OF PNP RANSAC ....")
            continue
    best.save(global_eval_path+'/'+folder_name+'/best.txt')

    
bestepocht, bestepochr, bestUnThres_t, bestepochrate, bestUnThres_rate = best.final_infor()
best.save(global_eval_path+'/'+folder_name+'/best.txt')

##

print("BEST T")
checkpoint =  global_eval_path + '/epoch_{:03d}.pth.tar'.format(bestepocht)
# checkpoint =  global_eval_path + '/epoch_' + str(990) +  '.pth.tar'
print(checkpoint)
evaler = Evaluator(experiment_name, model, test_criterion, checkpoint, test_set,
                    train_set, data_path, train_criterion, args.dataset, 
                    w_uncertainty={'state':True, 'threshold': bestUnThres_t}, 
                    eval_train=True, model_ver=args.model, folder_name = folder_name)
evaler.evaler()


print("BEST Rate")
checkpoint =  global_eval_path + '/epoch_{:03d}.pth.tar'.format(bestepochrate)
# checkpoint =  global_eval_path + '/epoch_' + str(990) +  '.pth.tar'
print(checkpoint)
evaler = Evaluator(experiment_name, model, test_criterion, checkpoint, test_set,
                    train_set, data_path, train_criterion, args.dataset, 
                    w_uncertainty={'state':True, 'threshold': bestUnThres_rate}, 
                    eval_train=True, model_ver=args.model, folder_name = folder_name)
evaler.evaler()

print("Best Epoch: {} ---  with Uncer_thres: {}".format(bestepochrate, bestUnThres_rate))


