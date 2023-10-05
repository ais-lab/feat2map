# this implementation is borrowed from 
# https://github.com/cvg/Hierarchical-Localization
# change logs: max_keypoints -> 2048 for trainin D2S
#               line 68->77: fill in blank space with zero descriptors 
#                               for training batchsize > 1


import copy
import numpy as np
import torch
import pycolmap
from .base_model import BaseModel



EPS = 1e-6


def sift_to_rootsift(x):
    x = x / (np.linalg.norm(x, ord=1, axis=-1, keepdims=True) + EPS)
    x = np.sqrt(x.clip(min=EPS))
    x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + EPS)
    return x


class SIFT(BaseModel):
    default_conf = {
        'num_octaves': 4,
        'octave_resolution': 3,
        'first_octave': 0,
        'edge_thresh': 10,
        'peak_thresh': 0.01,
        'upright': False,
        'root': True,
        'max_keypoints': 2048
    }
    required_inputs = ['image']

    def _init(self, conf):
        self.root = conf['root']
        self.max_keypoints = conf['max_keypoints']

        vlfeat_conf = copy.deepcopy(conf)
        vlfeat_conf.pop('name', None)
        vlfeat_conf.pop('root', None)
        vlfeat_conf.pop('max_keypoints', None)
        self.extract = lambda image: pycolmap.extract_sift(
            image, **vlfeat_conf
        )

    def _forward(self, data):
        image = data['image'].cpu().numpy()
        assert image.shape[1] == 1
        assert image.min() >= -EPS and image.max() <= 1 + EPS

        keypoints, scores, descriptors = self.extract(image[0, 0])
        keypoints = keypoints[:, : 2]  # Keep only x, y.

        if self.root:
            descriptors = sift_to_rootsift(descriptors)

        if self.max_keypoints != -1:
            # TODO: check that the scores from PyCOLMAP are 100% correct,
            # follow https://github.com/mihaidusmanu/pycolmap/issues/8
            indices = np.argsort(scores)[:: -1][: self.max_keypoints]
            keypoints = keypoints[indices, :]
            scores = scores[indices]
            descriptors = descriptors[indices, :]

            # thuan ... 
            if len(indices) < self.max_keypoints:
                # print(keypoints.shape)
                # print(scores.shape)
                # print(descriptors.shape)
                add_num = self.max_keypoints - len(indices)
                keypoints = np.concatenate((keypoints, np.zeros((add_num,2))), axis = 0)
                scores = np.concatenate((scores, np.zeros(add_num)), axis = 0)
                descriptors = np.concatenate((descriptors, np.zeros((add_num,128))), axis = 0)
            # end thuan .. 

        return {
            'keypoints': torch.from_numpy(keypoints)[None],
            'scores': torch.from_numpy(scores)[None],
            'descriptors': torch.from_numpy(descriptors.T)[None],
        }
