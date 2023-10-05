python train.py --dataset Cambridge --scene KingsCollege --config_file configs/configsV2.ini --model 2 --cudaid 0 --resume_optim 1 --unlabel 1 --unlabel_rate 0.5 --checkpoint 1214 --augment 1
python eval.py --dataset Cambridge --scene KingsCollege --config_file configs/configsV2.ini --model 2 --cudaid 0 --start_epoch 1424 --stop_epoch 1428 --idx_eval 3
 