import torch
import numpy as np
import torch.nn.functional as F

def transI_fusebn(kernel, bn):
    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    return kernel * ((gamma / std).reshape(-1, 1, 1, 1)), bn.bias - bn.running_mean * gamma / std

def transII_addbranch(kernels, biases):
    return sum(kernels), sum(biases)

def transIII_1x1_kxk(k1, b1, k2, b2, groups):
    if groups == 1:
        k = F.conv2d(k2, k1.permute(1, 0, 2, 3))      #
        if b1 !=0:
            b_hat = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3))
        return k,0
    else:
        k_slices = []
        b_slices = []
        k1_T = k1.permute(1, 0, 2, 3)
        k1_group_width = k1.size(0) // groups
        k2_group_width = k2.size(0) // groups
        for g in range(groups):
            k1_T_slice = k1_T[:, g*k1_group_width:(g+1)*k1_group_width, :, :]
            k2_slice = k2[g*k2_group_width:(g+1)*k2_group_width, :, :, :]
            k_slices.append(F.conv2d(k2_slice, k1_T_slice))
            b_slices.append((k2_slice * b1[g*k1_group_width:(g+1)*k1_group_width].reshape(1, -1, 1, 1)).sum((1, 2, 3)))
        k, b_hat = transIV_depthconcat(k_slices, b_slices)
    return k, b_hat + b2
    # return k, 0

def transIV_depthconcat(kernels, biases):
    return torch.cat(kernels, dim=0), torch.cat(biases)

def transV_avg(channels, kernel_size, groups):
    input_dim = channels // groups
    k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
    k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
    return k

#   This has not been tested with non-square kernels (kernel.size(2) != kernel.size(3)) nor even-size kernels
def transVI_multiscale(kernel, target_kernel_size):
    H_pixels_to_pad = (target_kernel_size - kernel.size(2)) // 2
    W_pixels_to_pad = (target_kernel_size - kernel.size(3)) // 2
    # return F.pad(kernel, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])
    return F.pad(kernel, [W_pixels_to_pad, W_pixels_to_pad, H_pixels_to_pad, H_pixels_to_pad])


# code for myself
def transVII_1x1_1xk_kx1(k1, b1, k2, b2, k3, b3, groups=2):
    # transform to concate [1xk, kx1], but need to split feature the into groups in advance
    # k1 is 1x1 convolution; k2: 1xk convolution; k3: kx1 convolution
    if b1 !=None:
        if b2 != None :
            k_slices = []
            b_slices = []
            k1_T = k1.permute(1, 0, 2, 3)
            k1_group_width = k1.size(0) // groups
            
            # for g in range(groups):
            k1_T_slice1 = k1_T[:, 0:k1_group_width, :, :]
            k_slices.append(F.conv2d(k2, k1_T_slice1))
            b_slices.append((k2 * b1[0:k1_group_width].reshape(1, -1, 1, 1)).sum((1, 2, 3)))

            k1_T_slice2 = k1_T[:, k1_group_width:2*k1_group_width, :, :]
            k_slices.append(F.conv2d(k3, k1_T_slice2))
            b_slices.append((k3 * b1[k1_group_width:2*k1_group_width].reshape(1, -1, 1, 1)).sum((1, 2, 3)))

            k, b_hat = transIV_depthconcat(k_slices, b_slices)
            return k, b_hat + torch.cat([b2, b3])
        else:
            k_slices = []
            b_slices = []
            k1_T = k1.permute(1, 0, 2, 3)
            k1_group_width = k1.size(0) // groups
            
            # for g in range(groups):
            k1_T_slice1 = k1_T[:, 0:k1_group_width, :, :]
            k_slices.append(F.conv2d(k2, k1_T_slice1))
            b_slices.append((k2 * b1[0:k1_group_width].reshape(1, -1, 1, 1)).sum((1, 2, 3)))

            k1_T_slice2 = k1_T[:, k1_group_width:2*k1_group_width, :, :]
            k_slices.append(F.conv2d(k3, k1_T_slice2))
            b_slices.append((k3 * b1[k1_group_width:2*k1_group_width].reshape(1, -1, 1, 1)).sum((1, 2, 3)))

            k, b_hat = transIV_depthconcat(k_slices, b_slices)
            return k, b_hat 
    else:
        k_slices = []
        k1_T = k1.permute(1, 0, 2, 3)
        k1_group_width = k1.size(0) // groups
        k1_T_slice1 = k1_T[:, 0:k1_group_width, :, :]
        k_slices.append(F.conv2d(k2, k1_T_slice1))
        k1_T_slice2 = k1_T[:, k1_group_width:2*k1_group_width, :, :]
        k_slices.append(F.conv2d(k3, k1_T_slice2))
        k = torch.cat(k_slices, dim=0)
        return k,  0
