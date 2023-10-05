#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:25:01 2022

@author: thuan
"""

import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))

def select_model(model_ver, config={}):
    if model_ver == 1:
        import models._3DFeatLoc_1 as v1 
        model = v1.MainModel()
        model_name ="model_1"
    elif model_ver == 2:
        import models._3DFeatLoc_2 as v1 
        model = v1.MainModel()
        model_name ="model_2"
    else:
        raise "Not implemented"
    
    return model, model_name