"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.optim as optim


class Optimizer:
    """
    Wrapper around torch.optim + learning rate
    """

    def __init__(self, params, method, base_lr, weight_decay, **kwargs):
        self.method = method
        self.base_lr = base_lr

        if self.method == 'sgd':
            self.lr_decay = kwargs.pop('lr_decay')
            self.lr_stepvalues = sorted(kwargs.pop('lr_stepvalues'))
            self.learner = optim.SGD(params, lr=self.base_lr,
                                     weight_decay=weight_decay, **kwargs)
        elif self.method == 'adam':
            print("OPTIMIZER: ---  adam")
            self.lr_decay = kwargs.pop('lr_decay')
            self.lr_stepvalues = sorted(kwargs.pop('lr_stepvalues'))
            self.learner = optim.Adam(params, lr=self.base_lr,
                                      weight_decay=weight_decay, **kwargs)

        elif self.method == 'rmsprop':
            self.learner = optim.RMSprop(params, lr=self.base_lr,
                                         weight_decay=weight_decay, **kwargs)

    def adjust_lr(self, epoch):
        if self.method not in ['sgd', 'adam']:
            return self.base_lr

        decay_factor = 1
        for s in self.lr_stepvalues:
            if epoch < s:
                break
            decay_factor *= self.lr_decay

        lr = self.base_lr * decay_factor

        for param_group in self.learner.param_groups:
            param_group['lr'] = lr
        return lr

    def mult_lr(self, f):
        for param_group in self.learner.param_groups:
            param_group['lr'] *= f
    def current_lr(self):
        for param_group in self.learner.param_groups:
            return param_group['lr']
    def update_new_lr(self, lr):
        for param_group in self.learner.param_groups:
            param_group['lr'] = lr
        return lr


def test_main(optimizer_configs):
    import sys
    import os.path as osp
    sys.path.append(osp.join(osp.dirname(__file__), ".."))
    from models._3DFeatLoc import MainModel
    n_epoch = 300
    optimizer_configs["lr_stepvalues"] = [k/4*n_epoch for k in range(1, 5)]
    model = MainModel()
    op = Optimizer(model.parameters(), **optimizer_configs)
    epochs = [i for i in range(n_epoch)]
    out_lrs = []
    for i in epochs: 
        out_lrs.append(op.adjust_lr(i))
    
    import matplotlib.pyplot as plt 
    plt.plot(epochs, out_lrs)
    plt.show()
if __name__ == "__main__":
    optimizer_configs = {
        'method': "adam",
        'base_lr': 0.1,
        'weight_decay': 0.0005,
        'lr_decay': 0.9,
    }
    test_main(optimizer_configs)
    
    
    