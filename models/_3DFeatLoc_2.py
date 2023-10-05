#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 12:11:13 2022

@author: thuan
"""

##############  Model version of this work "D2S": https://arxiv.org/abs/2307.15250
############## The graph attention is original in SuperGlue paper https://github.com/magicleap/SuperGluePretrainedNetwork

from torch import nn
import torch.nn.functional as F
import torch
from typing import List, Tuple
from copy import deepcopy

def MLP(channels:list):
    layers = []
    n_chnls = len(channels)
    for i in range(1, n_chnls):
        layers.append(nn.Conv1d(channels[i-1], channels[i], 
                                kernel_size=1, bias=True))
        if i < n_chnls-1:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)

    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, no_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(no_layers)])
        self.no_layers = no_layers

    def forward(self, desc: torch.Tensor):
        for i in range(self.no_layers):
            delta = self.layers[i](desc, desc)
            desc = desc + delta
        return desc




class MainModel(nn.Module):
    default_config = {
        'descriptor_dim': 256,
        'keypts_encoder': [64,128],
        'GNN_no_layers': 5,
    }

    def __init__(self, config={}):
        super().__init__()
        
        self.config = {**self.default_config,**config}

        self.gnn = AttentionalGNN(
            feature_dim=self.config['descriptor_dim'], no_layers=self.config['GNN_no_layers'])
        self.mapping = MLP([self.config['descriptor_dim'], 512, 1024, 1024, 512, 4])
       


    def forward(self, data):
        descpt = data['descriptors']
        out = self.gnn(descpt)
        out = self.mapping(out)
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
    in_data["image_size"] = torch.unsqueeze(in_data["image_size"], dim=0)
    
    print("in_data_descriptors shape: ", in_data["descriptors"].shape)
    
    target = data[0][1]
    target["p3Ds"] = torch.unsqueeze(target["p3Ds"], dim = 0)
    print("target.shape: ", target["p3Ds"].shape)
    model = MainModel()
    print("\nTotal parameters: {}".format(sum(p.numel() for p in model.parameters())))
    out = model(in_data)


if __name__ == "__main__":
    test_main()
    
  