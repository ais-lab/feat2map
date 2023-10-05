#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 14:38:24 2022

@author: thuan
"""


##############  Model version of this work "fast and lightweight" https://arxiv.org/abs/2212.01830

from torch import nn
import torch


class MainModel(nn.Module):
    default_config = {
        'descriptor_dim': 256,
    }

    def __init__(self, config={}):
        super().__init__()
        self.config = {**self.default_config,**config}
        
        self.feedfw = nn.Sequential(
            nn.Conv1d(self.config['descriptor_dim'], 512, 1), 
            nn.ReLU(),
            nn.Conv1d(512, 1024, 1),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
            nn.ReLU(),
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 4, 1)
            )

    def forward(self, data):
        descpt = data['descriptors']
        out = self.feedfw(descpt)

        return (out[:,:3,:], 1/(1+100*torch.abs(out[:,3,:])))


def test_main():
    # for testing
    import torch
    import sys
    import os.path as osp
    sys.path.append(osp.join(osp.dirname(__file__), ".."))
    from processing.dataloader import _3DFeatLoc_Loader
    # load data
    data = _3DFeatLoc_Loader(osp.join("../dataset/7scenes", "heads"))
    # feed forward
    in_data = data[0][0]
    in_data["descriptors"] = torch.unsqueeze(in_data["descriptors"], dim=0)
    
    print("in_data_descriptors shape: ", in_data["descriptors"].shape)
    
    target = data[0][1]
    target["p3Ds"] = torch.unsqueeze(target["p3Ds"], dim = 0)
    print("target.shape: ", target["p3Ds"].shape)
    model = MainModel()
    
    out = model(in_data)
    print("out.shape: {}".format(out.shape))
    print("A prediction feature coordinate is: {}".format(out[0,:,0]))
    
    from criterion import _3DFLCriterion
    loss = _3DFLCriterion()
    
    print("loss: {}".format(loss(out, target)))

if __name__ == "__main__":
    test_main()
    
  