#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 12:10:23 2022

@author: thuan
This code partially builds on MapNet: https://github.com/NVlabs/geomapnet
"""
import os
import sys
import os.path as osp
sys.path.append(osp.join(osp.dirname(__file__), ".."))
import configparser
import torch
import numpy as np
from visdom import Visdom
from processing.Logger import Logger, History
import time



def load_state_dict(model, state_dict):
    """
    Loads a state dict when the model has some prefix before the parameter names
    :param model: 
    :param state_dict: 
    :return: loaded model
    """
    model_names = [n for n,_ in model.named_parameters()]
    state_names = [n for n in state_dict.keys()]

    # find prefix for the model and state dicts from the first param name
    if model_names[0].find(state_names[0]) >= 0:
        model_prefix = model_names[0].replace(state_names[0], '')
        state_prefix = None
    elif state_names[0].find(model_names[0]) >= 0:
        state_prefix = state_names[0].replace(model_names[0], '')
        model_prefix = None
    else:
        print ('Could not find the correct prefixes between {:s} and {:s}'.\
               format(model_names[0], state_names[0]))
        raise KeyError

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        if state_prefix is None:
            k = model_prefix + k
        else:
            k = k.replace(state_prefix, '')
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)



class Trainer(object):
    def __init__(self, experiment_name, model, optimizer, train_criterion, config_file, 
                 train_set, val_set=None, checkpoint_file=None, resume_optim=False, 
                 val_criterion=None, args=None):
        self.args = args
        self.experiment_name = experiment_name
        self.resume_optim = resume_optim
        self.model = model
        self.train_criterion = train_criterion
        if val_criterion == None:
            self.val_criterion = self.train_criterion
        self.optimizer = optimizer
        # read the config
        settings = configparser.ConfigParser()
        with open(config_file, 'r') as f:
          settings.read_file(f)
        self.config = {}
        
        section = settings['training']
        self.config['batch_size'] = section.getint('batch_size')
        self.config['n_epochs'] = int(self.config['batch_size'] * section.getint('n_iters')/len(train_set))
        self.original_n_epochs = int(self.config['batch_size'] * section.getint('n_iters')/len(train_set))
        if self.resume_optim:
            if not self.args.unlabel:
                self.log_filename = "log2.txt"
                if self.args.checkpoint != 0:
                    self.config['n_epochs'] = self.args.checkpoint
                    self.original_n_epochs = self.args.checkpoint
                self.config['n_epochs'] += int(self.config['batch_size'] * section.getint('n_iters_add')/len(train_set))
            else:
                if self.args.checkpoint != 0:
                    self.original_n_epochs = self.args.checkpoint
                self.original_n_epochs = self.args.checkpoint
                self.log_filename = "log3.txt"
                self.config['n_epochs'] = self.args.checkpoint + self.config['batch_size']*int(section.getint('n_iters_unlabel_add')/len(train_set))
        else:
            self.log_filename = "log1.txt"
        
        self.config['do_val'] = section.getboolean('do_val')
        self.config['shuffle'] = section.getboolean('shuffle')
        self.config['seed'] = section.getint('seed')
        self.config['num_workers'] = section.getint('num_workers')
        self.config['snapshot'] = section.getint('snapshot')
        self.config['val_freq'] = section.getint('val_freq')
        self.config['cuda'] = torch.cuda.is_available()
        self.config["nGPUs"] = section.getint('n_GPUs')
        self.config['val_rate'] = section.getfloat('val_rate')
        
        section = settings['logging']
        self.config['log_visdom'] = section.getboolean('visdom')
        self.config['print_freq'] = section.getint('print_freq')

        section = settings['optimization']
        self.config['lr'] = section.getfloat('lr')
        self.config['lr_add'] = section.getfloat('lr_add')
        self.config['lr_add_unlabel'] = section.getfloat('lr_add_unlabel')
        self.config['lr_decay'] = section.getfloat('lr_decay')

        self.config["learn_from_augment_data"] = self.args.augment 
        self.config["learn_from_unlabel_data"] = self.args.unlabel

        self.logdir = osp.join(os.getcwd(), 'logs', self.experiment_name)
        if not osp.isdir(self.logdir):
            os.makedirs(self.logdir)
        
        self.logging = Logger(osp.join(self.logdir, self.log_filename))
        sys.stdout = self.logging
        
        # log all the command line options
        print ('---------------------------------------')
        print ('Experiment: {:s}'.format(self.experiment_name))
        for k, v in self.config.items():
            print ('{:s}: {:s}'.format(k, str(v)))
        print ('Using GPU {:d} / {:d}'.format(self.config["nGPUs"], torch.cuda.device_count()))
        print('Number Params is: {:.3f} M'.format(sum(p.numel() for p in model.parameters())/1000000))
        print ('---------------------------------------')
        
        # set random seed 
        torch.manual_seed(self.config['seed'])
        if self.config['cuda']:
            torch.cuda.manual_seed(self.config['seed'])
        
        # activate GPUs
        if self.config['cuda']:
            self.model.cuda()
            self.train_criterion.cuda()
            self.val_criterion.cuda()


        self.start_epoch = int(0) 
        if checkpoint_file:
            if osp.isfile(checkpoint_file):
                loc_func = None if self.config['cuda'] else lambda storage, loc: storage
                checkpoint = torch.load(checkpoint_file, map_location=loc_func)
                load_state_dict(self.model, checkpoint['model_state_dict'])
                if self.resume_optim:
                    self.optimizer.learner.load_state_dict(checkpoint['optim_state_dict'])
                    self.start_epoch = checkpoint['epoch']
                print ('Loaded checkpoint {:s} epoch {:d}'.format(checkpoint_file, checkpoint['epoch']))
        
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.config['batch_size'],
                                                        shuffle=self.config['shuffle'], num_workers=self.config['num_workers'], 
                                                        pin_memory=True)
        # note that shffle has been done from "tran_set" loader
        if self.config['do_val']: 
            self.val_loader = torch.utils.data.DataLoader(val_set, 
                                                          batch_size=self.config['batch_size'], shuffle=False,
                                                          num_workers=self.config['num_workers'], pin_memory=True)
        else:
            self.val_loader = None 
        
        if self.config['log_visdom']:
            # start plots 
            self.vis_env = experiment_name
            self.loss_win = "loss_win"
            self.vis = Visdom()
            self.vis.line(X=np.asarray([[self.start_epoch, self.start_epoch]]), Y=np.zeros((1,2)), win=self.loss_win,
                          opts={'legend': ['train_loss', 'val_loss'], 'xlabel': 'epochs',
                                'ylabel': 'loss'}, env=self.vis_env)
            
            self.l1_loss = "3dmap_loss"
            self.vis.line(X=np.asarray([self.start_epoch]), Y=np.zeros(1), win=self.l1_loss,
                          opts={'legend': ['3dmap_loss'], 'xlabel': 'epochs',
                                'ylabel': '3dmap_loss'}, env=self.vis_env)

            self.l2_loss = "reprj_loss"
            self.vis.line(X=np.asarray([self.start_epoch]), Y=np.zeros(1), win=self.l2_loss,
                          opts={'legend': ['reprj_loss'], 'xlabel': 'epochs',
                                'ylabel': 'reprj_loss'}, env=self.vis_env)
            self.lu_loss = "uncer_loss"
            self.vis.line(X=np.asarray([self.start_epoch]), Y=np.zeros(1), win=self.lu_loss,
                          opts={'legend': ['uncer_loss'], 'xlabel': 'epochs',
                                'ylabel': 'uncer_loss'}, env=self.vis_env)
            self.lp_loss = "pose_loss"
            self.vis.line(X=np.asarray([self.start_epoch]), Y=np.zeros(1), win=self.lp_loss,
                          opts={'legend': ['pose_loss'], 'xlabel': 'epochs',
                                'ylabel': 'pose_loss'}, env=self.vis_env)
            self.lr_win = 'lr_win'
            self.vis.line(X=np.asarray([self.start_epoch]), Y=np.zeros(1), win=self.lr_win,
                          opts={'legend': ['learning_rate'], 'xlabel': 'epochs',
                                'ylabel': 'lr'}, env=self.vis_env)
        
            
    def save_checkpoint(self, epoch):
        filename = osp.join(self.logdir, 'epoch_{:03d}.pth.tar'.format(epoch))
        checkpoint_dict =\
            {'epoch': epoch, 'model_state_dict': self.model.state_dict(),
             'optim_state_dict': self.optimizer.learner.state_dict()}
        torch.save(checkpoint_dict, filename)
    
    def train(self): 
        """
        Function that does the training and validation. 
        """
        start_time = time.time()
        for epoch in range(self.start_epoch, self.config['n_epochs']):
            # Validation 
            if self.config['do_val'] and ((epoch % self.config['val_freq']==0) or 
                                           (epoch == self.config['n_epochs']-1)):
                self.model.eval()
                val_loss = History()
                for batch_idx, (data, target) in enumerate(self.val_loader):
                    loss, _ = step_fwd(data, self.model, self.config['cuda'], target, self.val_criterion)
                    val_loss.update(loss)
                    if batch_idx % self.config['print_freq'] == 0:
                        print ('Val {:s}: Epoch {:d}\t' \
                               'Batch {:d}/{:d}\t' \
                                   'Loss {:f}' \
                                       .format(self.experiment_name, epoch, batch_idx, len(self.val_loader)-1, loss))
                        if self.config['log_visdom']:
                            self.vis.save(envs=[self.vis_env])
                print ('Val {:s}: Epoch {:d}, val_loss {:f}'.format(self.experiment_name, epoch, val_loss.average()))
            
            if self.config['log_visdom']:
                if self.config['do_val']:
                    self.vis.line(X=np.asarray([epoch]),
                                         Y=np.asarray([val_loss.average()]), win=self.loss_win, name='val_loss',
                                         update='append', env=self.vis_env)
                    self.vis.save(envs=[self.vis_env])
            
            # save checkpoint
            if epoch < (self.config['n_epochs']*0.88):
                if epoch % 20 == 0 or self.args.checkpoint != 0:
                    self.save_checkpoint(epoch)
                    print ('Epoch {:d} checkpoint saved for {:s}'.\
                       format(epoch, self.experiment_name))

            elif (epoch % self.config['snapshot'] == 0):
                self.save_checkpoint(epoch)
                print ('Epoch {:d} checkpoint saved for {:s}'.\
                       format(epoch, self.experiment_name))
                    
            # update new learning rate if resuming new learning 
            if self.resume_optim and epoch >= self.original_n_epochs:
                start_new_lr = self.config['lr_add_unlabel'] if self.args.unlabel else self.config['lr_add']
                lr = self.optimizer.update_new_lr(start_new_lr)
                self.optimizer.base_lr = start_new_lr
                self.resume_optim = False
            else:
                # adjust learning rate.
                ad_epoch = epoch if epoch < self.original_n_epochs else epoch - self.original_n_epochs
                lr = self.optimizer.adjust_lr(ad_epoch)

            if self.config['log_visdom']:
                self.vis.line(X=np.asarray([epoch]), Y=np.asarray([lr]),
                                     win=self.lr_win, name='learning_rate', update='append', env=self.vis_env)
                
            # training 
            self.model.train()
            start_time_print_freq = time.time()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                loss, _ = step_fwd(data, self.model, self.config['cuda'], target,
                                   self.train_criterion, self.optimizer, True, epoch)
                if batch_idx % self.config['print_freq'] == 0:
                    amount_time_print_freq = (time.time() - start_time_print_freq)*(len(self.train_loader)/self.config['print_freq'])*(self.config['n_epochs'] - epoch)/(60*60)
                    n_iter = epoch*len(self.train_loader) + batch_idx
                    epoch_count = float(n_iter)/len(self.train_loader)
                    print ('Train {:s}: Epoch {:d}\t Batch {:d}/{:d}\t Loss: {:.1f} Lmap: {:.1f} Lrepj: {:.1f} Lunc: {:.1f} lr: {:.9f} Expec_Finish_in: {:.1f} hours'.\
                                            format(self.experiment_name, epoch, batch_idx, len(self.train_loader)-1, loss[0], loss[1], loss[2], loss[3], lr, amount_time_print_freq))
                    if self.config['log_visdom']:
                        self.vis.line(X=np.asarray([epoch_count]),
                                             Y=np.asarray([loss[0]]), win=self.loss_win, name='train_loss',
                                             update='append', env=self.vis_env)
                        self.vis.line(X=np.asarray([epoch_count]), Y=np.asarray([loss[1]]),
                                     win=self.l1_loss, name='3dmap_loss', update='append', env=self.vis_env)
                        # if epoch_count > 2:
                        self.vis.line(X=np.asarray([epoch_count]), Y=np.asarray([loss[2]]),
                                     win=self.l2_loss, name='reprj_loss', update='append', env=self.vis_env)
                        
                        self.vis.line(X=np.asarray([epoch_count]), Y=np.asarray([loss[3]]),
                                     win=self.lu_loss, name='uncer_loss', update='append', env=self.vis_env)
                        if len(loss) > 4:
                            self.vis.line(X=np.asarray([epoch_count]), Y=np.asarray([loss[4]]),
                                         win=self.lp_loss, name='pose_loss', update='append', env=self.vis_env)
                    self.vis.save(envs=[self.vis_env])
                    start_time_print_freq = time.time()
        # save the last epoch. 
        epoch = self.config['n_epochs']
        self.save_checkpoint(epoch)
        print ('Epoch {:d} checkpoint saved'.format(epoch))
        print ("------ Total Training Time For This Part: ------ {}".format((time.time()-start_time)/(60*60)))
        if self.config['log_visdom']:
            self.vis.save(envs=[self.vis_env])
            
def step_fwd(data, model, cuda, target=None, criterion=None, optim=None, train=False, epoch=None):
    """
    A training/validation step."""
    
    if train: 
        assert criterion is not None
        assert target is not None
        assert optim is not None
        assert epoch is not None
    if cuda:
        for k,v in data.items():
            data[k] = data[k].cuda()
        if target is not None:
            for k,v in target.items():
                target[k] = target[k].cuda()

    with torch.set_grad_enabled(train):
        output = model(data)
    
    if criterion is not None:
        with torch.set_grad_enabled(train):
            loss = criterion(output, target, epoch)
        if train:
            optim.learner.zero_grad()
            loss[0].backward()
            optim.learner.step() 
        
        return [i.item() for i in loss], output
    else:
        return [0,0,0,0], output
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        