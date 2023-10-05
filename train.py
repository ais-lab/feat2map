#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 12:09:49 2022

@author: thuan
"""

from processing.dataloader import _3DFeatLoc_Loader, _3DFeatLoc_Loader_New
import argparse 
import os
from processing.optimizer import Optimizer
import configparser
from models import criterion
import json
from models.trainer import Trainer
import torch
from utils.utils import cal_train_val
from utils.select_model import select_model 

parser = argparse.ArgumentParser(description='Training script for 3DFeatLoc and'
                                 'its variants')

parser.add_argument('--dataset',type=str,default="7scenes",
                    help="name of scene to be trained")
parser.add_argument('--scene',type=str,default="heads",
                    help="name of scene to be trained")
parser.add_argument('--config_file',type=str,default="configs/configsV0.ini",
                    help="name of scene to be trained")
parser.add_argument('--model',type=int,default=2,
                    help="choose the model to be trained")
parser.add_argument('--checkpoint', type=int, help='checkpoint to resume from', 
                    default=0)
parser.add_argument('--resume_optim', type=bool, default=0,
                    help='Resume optimization (only effective if a checkpoint is given')
parser.add_argument('--augment', type=int, default=0, choices =[0,1],
                    help='apply data augementation or not')
parser.add_argument('--cudaid', type=int, 
                    default=1)
parser.add_argument('--use_mean', type=bool, default=0,
                    help='Resume optimization (only effective if a checkpoint is given')
parser.add_argument('--unlabel', type=bool, default=0,
                    help='Resume optimization (only effective if a checkpoint is given')
parser.add_argument('--unlabel_rate', type=float, default=1.0,
                    help='Resume optimization (only effective if a checkpoint is given')




args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cudaid)
settings = configparser.ConfigParser()
with open(args.config_file, 'r') as f:
    settings.read_file(f)


# load dataset
# data loader configs
dataset_dir = "/home/thuan/Desktop/GITHUB/FeatLoc_preparation/dataset/Hierarchical_Localization/datasets/"
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


train_set = _3DFeatLoc_Loader_New(os.path.join("dataset/" + args.dataset, args.scene), configs = data_loader_configs)


section = settings['training']
# seed = section.getint('seed')
batch_size = section.getint('batch_size')

num_epoch = int(batch_size * section.getint('n_iters')/len(train_set))
# if args.resume_optim:
num_epoch_add = int(batch_size * section.getint('n_iters_add')/len(train_set))
num_epoch_add_unlabel = int(batch_size * section.getint('n_iters_unlabel_add')/len(train_set))
if args.unlabel and args.checkpoint != 0:
    num_epoch = args.checkpoint

val_rate = section.getfloat('val_rate')


use_meanConfigs ={'use_mean': args.use_mean, 
'mean_path':os.path.join("dataset/" + args.dataset, args.scene, 'mean.txt')
}

model, model_name = select_model(args.model, use_meanConfigs)



# optimizer 
section = settings['optimization']
optim_config = {k: json.loads(v) for k,v in section.items() if k != 'opt'}
opt_method = section['opt']
lr = optim_config.pop('lr')
lr_decay = optim_config.pop('lr_decay')
weight_decay = optim_config.pop('weight_decay')



if not args.unlabel:
    num_times_decay = 3 if args.resume_optim else 5
    real_num_epoch = num_epoch_add if args.resume_optim else num_epoch
else: # for unlabel data
    num_times_decay = 3 
    real_num_epoch = num_epoch - args.checkpoint + num_epoch_add_unlabel

optimizer_configs = {
    'method': opt_method,
    'base_lr': lr,
    'weight_decay': weight_decay,
    'lr_decay': lr_decay,
    'lr_stepvalues': [k/5*real_num_epoch for k in range(1, 7)]
    }



if args.unlabel:
    train_criterion = criterion._3DFLCriterion_New({"coef_lrepr":optim_config['coef_lrepr_unlabel'], "start_lrepr": num_epoch}) 
else:
    train_criterion = criterion._3DFLCriterion_New({"coef_lrepr":optim_config['coef_lrepr'], "start_lrepr": num_epoch})

if data_loader_configs["unlabel"]:# or args.dataset == "Cambridge":
    print("------------------------- Training with unlabel data, rate of {} ----------------------".format(data_loader_configs['unlabel_rate']))



param_list = [{'params': model.parameters()}]
optimizer = Optimizer(params = param_list, **optimizer_configs)



# trainer
config_name = args.config_file.replace("configs/", "")
config_name = config_name.split('.')[0]

experiment_name = '{:s}_{:s}_{:s}_{:s}'.format(args.dataset, args.scene, config_name, model_name)


if val_rate != 0:
    train_set, val_set = torch.utils.data.random_split(train_set, cal_train_val(len(train_set), val_rate))
else:
    val_set = None
global_eval_path = 'logs/' + experiment_name
if args.resume_optim: 
    
    if args.checkpoint == 0:
        checkpoint = num_epoch
    else:
        checkpoint = args.checkpoint 
    checkpoint_file = global_eval_path + '/epoch_{:03d}.pth.tar'.format(checkpoint)
    print(checkpoint_file)
    assert os.path.isfile(checkpoint_file)
else:
    if args.checkpoint != 0:
        checkpoint_file = global_eval_path + '/epoch_{:03d}.pth.tar'.format(args.checkpoint)
    else:
        checkpoint_file = None

trainer = Trainer(experiment_name, model, optimizer, train_criterion, args.config_file, train_set, 
                  val_set, checkpoint_file=checkpoint_file, resume_optim=args.resume_optim, 
                  val_criterion=None, args = args)
trainer.train()


