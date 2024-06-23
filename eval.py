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


def main():

    parser = argparse.ArgumentParser(description='Training script for 3DFeatLoc and'
                                    'its variants')

    parser.add_argument('--dataset',type=str,default="7scenes",
                        help="name of the dataset")
    parser.add_argument('--scene',type=str,default="chess",
                        help="name of scene")
    parser.add_argument('--config_file',type=str,default="configs/configsV2.ini",
                        help="configs path")
    parser.add_argument('--model',type=int,default=2,
                        help="choose the model to be trained")
    parser.add_argument('--eval_train', type=int, 
                        default=0)
    parser.add_argument('--checkpoint', type=str, 
                        default=None)
    parser.add_argument('--cudaid', type=int, 
                        default=1)
    parser.add_argument('--uncer', type=bool, 
                        default=True)
    parser.add_argument('--uncer_thres', type=float, 
                        default=0.5)


    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cudaid)
    config_name = args.config_file.replace("configs/", "")
    config_name = config_name.split('.')[0]

    model, model_name = select_model(args.model)

    # criterion 
    train_criterion = criterion._3DFLCriterion()
    test_criterion = None

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
        "unlabel": False, 
        "unlabel_rate": 0,
        "augment": False,
        "imgdata_dir": data_dir,
    }
    train_set_for_cal_nepoch = _3DFeatLoc_Loader_New(os.path.join("dataset/" + args.dataset, args.scene), configs = data_loader_configs)


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

    thres_unc = args.uncer_thres
    uncertainty_config = {'state':args.uncer, 'threshold':thres_unc}
    if uncertainty_config['state']:
        print('testing with UNCERTAINTY, threshold = {}'.format(uncertainty_config['threshold']))

    eval_train = args.eval_train

    print("Working on scene: {}, is_uncer: {}, uncer_thres: {}".format(args.scene, args.uncer, thres_unc))


    evaler = Evaluator(args.scene, model, test_criterion, args.checkpoint, test_set,
                        train_set, data_path, train_criterion, args.dataset, eval_train = args.eval_train, 
                        w_uncertainty=uncertainty_config, model_ver=args.model)
    evaler.evaler()


if __name__ == '__main__':
    main()

