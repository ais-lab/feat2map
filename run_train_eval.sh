#!/bin/bash
# python train.py --dataset 7scenes --scene office --config_file configs/configsV2.ini --model 2 --cudaid 0 --augment 0
# python train.py --dataset 7scenes --scene office --config_file configs/configsV2.ini --model 2 --cudaid 0 --resume_optim 1 --augment 0
python eval.py --dataset 7scenes --scene office --config_file configs/configsV2.ini --model 2 --cudaid 0 --augment 0


