from API.utils import create_random_shape_with_random_motion
import numpy as np
import os
mask_all = []
for i in range(10000):
    masks = create_random_shape_with_random_motion(
                10, imageHeight=64, imageWidth=64
            )
    masks = np.stack([np.expand_dims(x, 2) for x in masks], axis=0)/ 255.0
    mask_all.append(masks)
mask_all = np.stack(mask_all,axis=0)
np.save('./data/moving_mnist/mask.npy', mask_all)


