from models.pipeline import Pipeline
import cv2
import torch
import util.config as utilcfg
from omegaconf import OmegaConf
import numpy as np
import poselib

def getPoint3D_from_modeloutput(points3D, threshold=0.5):
    '''
    get uncertainty and remove point with high uncertainty
    args:
        points3D: numpy array (1, 4, N)
    return: points3D (N, 3)
    '''
    points3D = np.squeeze(points3D.detach().cpu().numpy())
    uncertainty = 1/(1+100*np.abs(points3D[3,:]))
    points3D = points3D[:3,:]
    uncertainty = [True if tmpc >= threshold else False for tmpc in uncertainty]
    points3D = points3D.T[uncertainty,:]
    return points3D, uncertainty
  
def camera_pose_estimation(output, camera_params=None, uncertainty_point=0.5):
    p3ds_, point_uncer = getPoint3D_from_modeloutput(output['points3D'], uncertainty_point)
    p3ds = [i for i in p3ds_]
    p2ds = output['keypoints'][0].detach().cpu().numpy() + 0.5 # COLMAP
    p2ds = p2ds[point_uncer,:]
    p2ds = [i for i in p2ds]
    if camera_params is not None:
        pass
    else:
        camera_model = "SIMPLE_RADIAL"
        poselibcamera = {'model': camera_model, 'width': 1080, 'height': 1920, 'params': np.array([1434.8912946839755, 
                                                                                                   540, 
                                                                                                   960,
                                                                                                   0.036490905769671532])}
    
    pose, _ = poselib.estimate_absolute_pose(p2ds, p3ds, poselibcamera, {'max_reproj_error': 12.0}, {})
    print(f"x:{pose.t[0]} \n y:{pose.t[1]} \n z:{pose.t[2]} \n qw:{pose.q[0]} \n qx:{pose.q[1]} \n qy:{pose.q[2]} \n qz:{pose.q[3]}")
    return pose
  
def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f'Cannot read image {path}.')
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image
  
def resize_image(image, size, interp):
    if interp.startswith('cv2_'):
        interp = getattr(cv2, 'INTER_'+interp[len('cv2_'):].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    else:
        raise ValueError(
            f'Unknown interpolation {interp}.')
    return resized
  
def map_dict_to_torch(data, device):
    for k, v in data.items():
        if isinstance(v, str):
            continue
        elif isinstance(v, np.ndarray):
            data[k] = torch.from_numpy(v).float().to(device)
        elif isinstance(v, float):
            data[k] = torch.tensor(v).float().to(device)
        else:
            raise ValueError("Error! Type {0} not supported.".format(type(v)))
    return data
  
if __name__ == "__main__":
    image_path ="custom_dataset/images/image1000000.png"
    # permanent path for model params 
    model_path = "logs/custom_d2s/custom_custom_d2s.pth.tar"
    path_to_eval_cfg = "logs/custom_d2s/config.yaml"
    cfg = utilcfg.load_config(path_to_eval_cfg, default_path='cfgs/default.yaml')
    cfg = OmegaConf.create(cfg)
    # setup model and load pre-trained weight
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Pipeline(cfg)
    model.load_checkpoint(model_path)
    model.to(device)
    # load data 
    resize_max = cfg.point2d.detector.preprocessing.resize_max
    image = read_image(image_path, grayscale=True)
    ori_size = image.shape[:2][::-1]
    
    scale = 1.0
    if max(ori_size) > resize_max:
        scale = resize_max / max(ori_size)
        aug_size = image.shape[::-1]
        size_new = tuple(int(round(x*scale)) for x in aug_size)
        image = resize_image(image, size_new, cfg.point2d.detector.preprocessing.interpolation)
        
    image = image.astype(np.float32)
    image = image[None][None]
    image = image / 255.
    original_size = np.array(ori_size)
    data = {
            'image': image,
            'original_size': original_size,
            'scale': scale,
        }
    data = map_dict_to_torch(data, device)
    data['keypoints'] = []
    pred = model(data)
    pose = camera_pose_estimation(pred)
