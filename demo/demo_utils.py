from pathlib import Path
import time
from collections import OrderedDict
from threading import Thread
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib
import copy 
import pandas as pd
import os
import time
import pycolmap

matplotlib.use('Agg')


class VideoStreamer:
    """ Class to help process image streams. Four types of possible inputs:"
        1.) USB Webcam.
        2.) An IP camera
        3.) A directory of images (files in directory matching 'image_glob').
        4.) A video file, such as an .mp4 or .avi file.
    """
    def __init__(self, basedir, resize, skip, image_glob, max_length=1000000):
        self._ip_grabbed = False
        self._ip_running = False
        self._ip_camera = False
        self._ip_image = None
        self._ip_index = 0
        self.cap = []
        self.camera = True
        self.video_file = False
        self.listing = []
        self.resize = resize
        self.interp = cv2.INTER_AREA
        self.i = 0
        self.skip = skip
        self.max_length = max_length
        if isinstance(basedir, int) or basedir.isdigit():
            print('==> Processing USB webcam input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(int(basedir))
            self.listing = range(0, self.max_length)
        elif basedir.startswith(('http', 'rtsp')):
            print('==> Processing IP camera input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.start_ip_camera_thread()
            self._ip_camera = True
            self.listing = range(0, self.max_length)
        elif Path(basedir).is_dir():
            print('==> Processing image directory input: {}'.format(basedir))
            self.listing = list(Path(basedir).glob(image_glob[0]))
            for j in range(1, len(image_glob)):
                image_path = list(Path(basedir).glob(image_glob[j]))
                self.listing = self.listing + image_path
            self.listing.sort()
            self.listing = self.listing[::self.skip]
            self.max_length = np.min([self.max_length, len(self.listing)])
            if self.max_length == 0:
                raise IOError('No images found (maybe bad \'image_glob\' ?)')
            self.listing = self.listing[:self.max_length]
            self.camera = False
        elif Path(basedir).exists():
            print('==> Processing video input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.listing = range(0, num_frames)
            self.listing = self.listing[::self.skip]
            self.video_file = True
            self.max_length = np.min([self.max_length, len(self.listing)])
            self.listing = self.listing[:self.max_length]
        else:
            raise ValueError('VideoStreamer input \"{}\" not recognized.'.format(basedir))
        if self.camera and not self.cap.isOpened():
            raise IOError('Could not read camera')

    def load_image(self, impath):
        """ Read image as grayscale and resize to img_size.
        Inputs
            impath: Path to input image.
        Returns
            grayim: uint8 numpy array sized H x W.
        """
        grayim = cv2.imread(impath, 0)
        rgbim = cv2.imread(impath)
        if grayim is None:
            raise Exception('Error reading image %s' % impath)
        w, h = grayim.shape[1], grayim.shape[0]
        w_new, h_new = process_resize(w, h, self.resize)
        grayim = cv2.resize(
            grayim, (w_new, h_new), interpolation=self.interp)
        rgbim = cv2.resize(
            rgbim, (w_new, h_new), interpolation=self.interp)
        return grayim, rgbim

    def next_frame(self):
        """ Return the next frame, and increment internal counter.
        Returns
             image: Next H x W image.
             status: True or False depending whether image was loaded.
        """

        if self.i == self.max_length:
            return (None, None, None, False)
        if self.camera:

            if self._ip_camera:
                #Wait for first image, making sure we haven't exited
                while self._ip_grabbed is False and self._ip_exited is False:
                    time.sleep(.001)

                ret, image = self._ip_grabbed, self._ip_image.copy()
                if ret is False:
                    self._ip_running = False
            else:
                ret, image = self.cap.read()
            if ret is False:
                print('VideoStreamer: Cannot get image from camera')
                return (None, None, None, False)
            w, h = image.shape[1], image.shape[0]
            if self.video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])

            w_new, h_new = process_resize(w, h, self.resize)
            image = cv2.resize(image, (w_new, h_new),
                               interpolation=self.interp)
            rgb_img = copy.deepcopy(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_file = str(self.listing[self.i])
            image, rgb_img = self.load_image(image_file)
        self.i = self.i + 1
        return (image, rgb_img, image_file, True)

    def start_ip_camera_thread(self):
        self._ip_thread = Thread(target=self.update_ip_camera, args=())
        self._ip_running = True
        self._ip_thread.start()
        self._ip_exited = False
        return self

    def update_ip_camera(self):
        while self._ip_running:
            ret, img = self.cap.read()
            if ret is False:
                self._ip_running = False
                self._ip_exited = True
                self._ip_grabbed = False
                return

            self._ip_image = img
            self._ip_grabbed = ret
            self._ip_index += 1
            #print('IPCAMERA THREAD got frame {}'.format(self._ip_index))

def process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def make_plot_fast(image, kpts, uncertainty, show_keypoints=True):

    out = image 

    if show_keypoints:
        kpts = np.round(kpts).astype(int) 
        lime = (0, 255, 0)
        red = (255, 0, 0)
        i = 0
        for x, y in kpts:
            if uncertainty[i]:
                    cv2.circle(out, (x, y), 2, lime, -1, lineType=cv2.LINE_AA)
            else:
                    cv2.circle(out, (x, y), 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            i += 1

    return out



class PnP_Pose:
    def __init__(self, dataset="7scenes", max_error_px = 12):
        self.dataset = dataset
        self.max_error_px = max_error_px
        self.get_camera()
    def get_camera(self):
        if self.dataset == "7scenes":
            self.camera = {'model': 'SIMPLE_PINHOLE', 
                  'width': 640, 'height': 480, 
                  'params':  np.array([525.0, 320.0, 240.])} 
        elif self.dataset == "AIS":
            self.camera = {'model': 'SIMPLE_PINHOLE', 
                  'width': 1920, 'height': 1080, 
                  'params':  np.array([1810.117948440442, 960.0, 540.0, -0.4134062351224796])}
        elif self.dataset == "Cambridge":
            self.camera = {'model': 'SIMPLE_PINHOLE', 
                  'width': 1920, 'height': 1080, 
                  'params':  np.array([1670.6999999999998, 960.0, 540.0, 0.0])}
        elif self.dataset == "indoor6":
            self.camera = {'model': 'SIMPLE_PINHOLE', 
                  'width': 1280, 'height': 720, 
                  'params':  np.array([904.096498254811, 640.0, 360.0, -0.0504315987533166])}
        elif self.dataset == "BKC":
            self.camera = {'model': 'SIMPLE_PINHOLE', 
                  'width': 1920, 'height': 1080, 
                  'params':  np.array([1840.037672, 960.000000, 540.000000, 0.031604])}

    def pnp(self, points2D, points3D):
        ans = pycolmap.absolute_pose_estimation(points2D, points3D, self.camera, self.max_error_px)
        pose = np.zeros((1,7))
        pose[0,:3] = ans['tvec']
        pose[0,3:] = ans['qvec']
        return pose, ans['num_inliers']

class Timer:
    def __init__(self, subject = "RANSAC PnP"):
        self.list_time = []
        self.subject = subject
    def start(self):
        self.start_time = time.time()
    def stop(self):
        self.list_time.append(time.time() - self.start_time)
    def eval(self):
        print("------- mean of {} is {} --------".format(self.subject, sum(self.list_time)/len(self.list_time)))
    def infor(self, nu_feats, ninliers):
        period = time.time() - self.start_time
        print("---- FPS: {:.3f} --#feats: {} --#inliers: {} ----".format(1/period, nu_feats, ninliers))


class GroundTruth():
    """docstring for GroundTruth"""
    def __init__(self, root, dataset, scene):
        self.root = root
        self.dataset = dataset
        self.scene = scene
        self.index = 1 if self.dataset == "indoor6" else 2
        self.get_infor()

    def get_infor(self):
        self.readme_path = os.path.join(self.root, "dataset", self.dataset, self.scene, "test", "readme.txt")
        self.infor = pd.read_csv(self.readme_path, header=None, sep = " ")
        self.dict_infor = {}
        for i in range(len(self.infor)):
            name = self.infor.iloc[i,0]
            self.dict_infor[name] = self.infor.iloc[i,2:9].to_numpy()
    def get_gt_pose(self, image_file):
        name  = '/'.join(image_file.split('/')[-self.index:])
        try:
            pose = self.dict_infor[name]
        except:
            print("Cannot find {} in the GroundTruth".format(name))
            return None, False
        return pose.reshape(1, 7), True

class DSACresults():
    """docstring for GroundTruth"""
    def __init__(self, root="/home/thuan/Desktop/westwing_seq3.txt"):
        self.get_infor(root)

    def get_infor(self, root):
        self.infor = pd.read_csv(root, header=None, sep = " ")
        self.dict_infor = {}
        for i in range(len(self.infor)):
            name = self.infor.iloc[i,0]
            q = self.infor.iloc[i,1:5].to_numpy()
            t = self.infor.iloc[i,5:8].to_numpy()
            self.dict_infor[name] = np.concatenate((t,q))

    def get_pose(self, image_file):
        name  = '/'.join(image_file.split('/')[-2:])
        try:
            pose = self.dict_infor[name]
        except:
            print("Cannot find {} in the GroundTruth".format(name))
            return None, False
        return pose.reshape(1, 7), True