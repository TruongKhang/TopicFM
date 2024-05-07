import io
from loguru import logger

import cv2
import numpy as np
import h5py
import torch
from numpy.linalg import inv

# --- DATA IO ---


def cv_imread(path, gray=False, augment_fn=None):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if augment_fn is not None:
        image = augment_fn(image)
    if gray:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def get_resized_wh(w, h, resize=None):
    if (resize is not None): # and (max(h,w) > resize):  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new


def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, :inp.shape[1], :inp.shape[2]] = inp
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
            mask[:, :inp.shape[1], :inp.shape[2]] = True
    else:
        raise NotImplementedError()
    return padded, mask


# --- MEGADEPTH ---

def read_megadepth_gray(path, resize=None, df=None, padding=False, augment_fn=None, geometric_aug=None):
    """
    Args:
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read image
    image = cv_imread(path, gray=True, augment_fn=augment_fn) # imread_gray(path, augment_fn)
    H_mat = None
    if geometric_aug is not None:
        image = torch.from_numpy(image).float() / 255
        image, H_mat = geometric_aug(image[None, None])
        image = image[0, 0].numpy() * 255

    # resize image
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = get_resized_wh(w, h, resize)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)

    image = cv2.resize(image, (w_new, h_new))
    scale = torch.tensor([w/w_new, h/h_new], dtype=torch.float)

    if padding:  # padding
        pad_to = resize #max(h_new, w_new)
        image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
    else:
        mask = None

    image = torch.from_numpy(image).float()[None] / 255  # (h, w) -> (1, h, w) and normalized
    mask = torch.from_numpy(mask) if mask is not None else None

    return image, mask, scale, H_mat


def read_megadepth_color(path, resize=None, df=None, padding=False, augment_fn=None, geometric_aug=None):
    """
    Args:
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read image
    image = cv_imread(path, gray=False, augment_fn=augment_fn) # imread_gray(path, augment_fn)
    H_mat = None
    if geometric_aug is not None:
        image = torch.from_numpy(image).float() / 255
        image = image.permute(2, 0, 1)
        image, H_mat = geometric_aug(image[None])
        image = image[0].permute(1, 2, 0)
        image = image.numpy() * 255

    # resize image
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = get_resized_wh(w, h, resize)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)

    image = cv2.resize(image, (w_new, h_new))
    scale = torch.tensor([w/w_new, h/h_new], dtype=torch.float)

    if padding:  # padding
        pad_to = resize #max(h_new, w_new)
        image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
    else:
        mask = None

    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255  # (h, w) -> (3, h, w) and normalized
    mask = torch.from_numpy(mask) if mask is not None else None

    return image, mask, scale, H_mat


def read_megadepth_depth(path, pad_to=None):
    depth = np.array(h5py.File(path, 'r')['depth'])
    if pad_to is not None:
        depth, _ = pad_bottom_right(depth, pad_to, ret_mask=False)
    depth = torch.from_numpy(depth).float()  # (h, w)
    return depth


# --- ScanNet ---

def read_scannet_gray(path, resize=(640, 480), augment_fn=None):
    """
    Args:
        resize (tuple): align image to depthmap, in (w, h).
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read and resize image
    image = cv_imread(path, gray=True, augment_fn=augment_fn)
    image = cv2.resize(image, resize)

    # (h, w) -> (1, h, w) and normalized
    image = torch.from_numpy(image).float()[None] / 255
    return image


def read_scannet_color(path, resize=(640, 480), augment_fn=None):
    """
    Args:
        resize (tuple): align image to depthmap, in (w, h).
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read and resize image
    image = cv_imread(path, gray=False, augment_fn=augment_fn)
    image = cv2.resize(image, resize)

    # (h, w) -> (1, h, w) and normalized
    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255
    return image


# ---- evaluation datasets: HLoc, Aachen, InLoc

def read_img_gray(path, resize=None, down_factor=16):
    # read and resize image
    image = cv_imread(path, gray=True, augment_fn=None)
    w, h = image.shape[1], image.shape[0]
    if (resize is not None) and (max(h, w) > resize):
        scale = float(resize / max(h, w))
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    else:
        w_new, h_new = w, h
    w_new, h_new = get_divisible_wh(w_new, h_new, down_factor)
    image = cv2.resize(image, (w_new, h_new))

    # (h, w) -> (1, h, w) and normalized
    image = torch.from_numpy(image).float()[None] / 255
    scale = torch.tensor([w / w_new, h / h_new], dtype=torch.float)
    return image, scale

def read_img_color(path, resize=None, down_factor=16):
    # read and resize image
    image = cv_imread(path, gray=False, augment_fn=None)
    w, h = image.shape[1], image.shape[0]
    if (resize is not None) and (max(h, w) > resize):
        scale = float(resize / max(h, w))
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    else:
        w_new, h_new = w, h
    w_new, h_new = get_divisible_wh(w_new, h_new, down_factor)
    image = cv2.resize(image, (w_new, h_new))

    # (h, w) -> (1, h, w) and normalized
    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255
    scale = torch.tensor([w / w_new, h / h_new], dtype=torch.float)
    return image, scale


def read_scannet_depth(path):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    depth = depth / 1000
    depth = torch.from_numpy(depth).float()  # (h, w)
    return depth


def read_scannet_pose(path):
    """ Read ScanNet's Camera2World pose and transform it to World2Camera.
    
    Returns:
        pose_w2c (np.ndarray): (4, 4)
    """
    cam2world = np.loadtxt(path, delimiter=' ')
    world2cam = inv(cam2world)
    return world2cam


def read_scannet_intrinsic(path):
    """ Read ScanNet's intrinsic matrix and return the 3x3 matrix.
    """
    intrinsic = np.loadtxt(path, delimiter=' ')
    return intrinsic[:-1, :-1]
