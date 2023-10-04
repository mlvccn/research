from PIL import Image
import numpy as np
import os
from glob import glob
import torchvision.transforms as transforms
# from skimage.measure import find_contours

# color palette, maping class {0,1,2,...} to some specific colors
COLORS = [
    [0.000, 0.447, 0.741], 
    [0.850, 0.325, 0.098], 
    [0.929, 0.694, 0.125], 
    [0.494, 0.184, 0.556], 
    [0.466, 0.674, 0.188], 
    [0.301, 0.745, 0.933]
] * 100 * 255
# 1.蓝色； 2.橙色；3.黄色；4.紫色；5.绿色；6.淡蓝色

def apply_mask(image, mask, color, alpha=0.5):
    """_summary_

    Args:
        image (numpy array): [h, w, 3]
        mask (numpy array): _description_
        color (list): [r,g,b]
        alpha (float, optional): _description_. Defaults to 0.5.
    Returns:
        numpy array: _description_
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def draw_countour(image, mask, color):
    """_summary_

    Args:
        image (_type_): _description_
        mask (_type_): _description_
        color (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    return image



im_mean = (124, 116, 104)

im_normalization = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )

inv_im_trans = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225])

def tensor_to_numpy(image):
    image_np = (image.numpy() * 255).astype('uint8')
    return image_np

def transpose_np(x):
    return np.transpose(x, [1,2,0])

def detach_to_cpu(x):
    return x.detach().cpu()

def tensor_to_im(x):
    x = detach_to_cpu(x)
    x = inv_im_trans(x).clamp(0, 1)
    x = tensor_to_numpy(x)
    x = transpose_np(x)
    return x

