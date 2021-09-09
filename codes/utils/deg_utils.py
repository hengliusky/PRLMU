import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.util import imresize
import random
from scipy.io import loadmat
from torch.autograd import Variable


def DUF_downsample(x, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code

    Args:
        x (Tensor, [B, T, C, H, W]): frames to be downsampled.
        scale (int): downsampling factor: 2 | 3 | 4.
    """

    assert scale in [2, 3, 4], "Scale [{}] is not supported".format(scale)

    def gkern(kernlen=13, nsig=1.6):
        import scipy.ndimage.filters as fi

        inp = np.zeros((kernlen, kernlen))
        # set element at the middle to one, a dirac delta
        inp[kernlen // 2, kernlen // 2] = 1
        # gaussian-smooth the dirac, resulting in a gaussian filter mask
        return fi.gaussian_filter(inp, nsig)

    B, T, C, H, W = x.size()
    x = x.view(-1, 1, H, W)
    pad_w, pad_h = 6 + scale * 2, 6 + scale * 2  # 6 is the pad of the gaussian filter
    r_h, r_w = 0, 0
    if scale == 3:
        r_h = 3 - (H % 3)
        r_w = 3 - (W % 3)
    x = F.pad(x, [pad_w, pad_w + r_w, pad_h, pad_h + r_h], "reflect")

    gaussian_filter = (
        torch.from_numpy(gkern(13, 0.4 * scale)).type_as(x).unsqueeze(0).unsqueeze(0)
    )
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    x = x.view(B, T, C, x.size(2), x.size(3))
    return x


def PCA(data, k=2):
    X = torch.from_numpy(data)
    X_mean = torch.mean(X, 0)
    X = X - X_mean.expand_as(X)
    U, S, V = torch.svd(torch.t(X))
    return U[:, :k]  # PCA matrix


def batch_random_size_kernel(sigma, batch_size=8, batch=1, tensor=True, ):
    kernels = []
    sigs = []
    sizes = []
    for i in range(batch_size):
        # sig = (np.random.uniform(sig_min, sig_max, (1, 1, 1)))
        sig = random.choice(sigma)
        if sig == 2:
            l = 11

        elif sig == 3:
            l = 17
            # l = 21
        elif sig == 4:
            l = 23

        sizes.append(l)
        ax = np.arange(-l // 2 + 1.0, l // 2 + 1.0)
        xx, yy = np.meshgrid(ax, ax)
        xx = xx[None].repeat(batch, 0)
        yy = yy[None].repeat(batch, 0)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sig ** 2))
        kernel = kernel / np.sum(kernel, (1, 2), keepdims=True)
        kernels.append(torch.FloatTensor(kernel) if tensor else kernel)
        # plt.imshow(kernel.squeeze(0))
        # plt.pause(1)
        sigs.append(sig)
    # print('input:', sigs, sizes)
    return kernels, sigs, sizes


def batch_stable_kernel(label_sigma, batch=1, tensor=True):
    kernels = []
    sizes = []
    for sig in (label_sigma):
        # sig = sig + 2
        if sig == 2:
            # l = 11
            l = 21
        elif sig == 3:
            l = 17
        elif sig == 4:
            l = 23
        else:
            print('no such sigma')
            break
        sizes.append(l)

        ax = np.arange(-l // 2 + 1.0, l // 2 + 1.0)
        xx, yy = np.meshgrid(ax, ax)
        xx = xx[None].repeat(batch, 0)
        yy = yy[None].repeat(batch, 0)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sig ** 2))
        kernel = kernel / np.sum(kernel, (1, 2), keepdims=True)
        kernels.append(torch.FloatTensor(kernel) if tensor else kernel)
    return kernels, sizes


def b_Bicubic(variable, scale):
    B, C, H, W = variable.size()
    H_new = int(H / scale)
    W_new = int(W / scale)
    tensor_v = variable.view((B, C, H, W))
    re_tensor = imresize(tensor_v, 1 / scale)
    return re_tensor


def random_batch_noise(batch, high, rate_cln=1.0):
    noise_level = np.random.uniform(size=(batch, 1)) * high
    noise_mask = np.random.uniform(size=(batch, 1))
    noise_mask[noise_mask < rate_cln] = 0
    noise_mask[noise_mask >= rate_cln] = 1
    return noise_level * noise_mask


def b_GaussianNoising(tensor, sigma, mean=0.0, noise_size=None, min=0.0, max=1.0):
    print(1)
    if noise_size is None:
        size = tensor.size()
    else:
        size = noise_size
    noise = torch.mul(
        torch.FloatTensor(np.random.normal(loc=mean, scale=1.0, size=size)),
        sigma.view(sigma.size() + (1, 1)),
    ).to(tensor.device)
    return torch.clamp(noise + tensor, min=min, max=max)

def b_GaussianNoising(tensor, noise_high, mean=0.0, noise_size=None, min=0.0, max=1.0):
    print(2)
    if noise_size is None:
        size = tensor.size()
    else:
        size = noise_size
    noise = torch.FloatTensor(
        np.random.normal(loc=mean, scale=noise_high, size=size)
        ).to(tensor.device)
    return torch.clamp(noise + tensor, min=min, max=max)


class BatchBlur(object):
    def __init__(self, l):
        self.l = l
        # print(self.l)
        # self.pad = nn.ZeroPad2d(l // 2)

    def __call__(self, input, kernels):

        blur_img = []
        for i in range(len(self.l)):
            if self.l[i] % 2 == 1:
                self.pad = (self.l[i] // 2, self.l[i] // 2, self.l[i] // 2, self.l[i] // 2)
            else:
                self.pad = (self.l[i] // 2, self.l[i] // 2 - 1, self.l[i] // 2, self.l[i] // 2 - 1)

            B, C, H, W = input[i:i + 1, :, :, :].size()
            # print('hr:', input.shape)
            pad = F.pad(input[i:i + 1, :, :, :], self.pad, mode='reflect')
            H_p, W_p = pad.size()[-2:]
            # print(kernels[i].shape)
            if len(kernels[i].size()) == 2:
                input_CBHW = pad.view((C * B, 1, H_p, W_p))
                kernel_var = kernels[i].contiguous().view((1, 1, self.l[i], self.l[i]))
                blur_img.append(F.conv2d(input_CBHW, kernel_var, padding=0).view((B, C, H, W)))
            else:
                input_CBHW = pad.view((1, C * B, H_p, W_p))
                kernel_var = (
                    kernels[i].contiguous()
                        .view((B, 1, self.l[i], self.l[i]))
                        .repeat(1, C, 1, 1)
                        .view((B * C, 1, self.l[i], self.l[i]))
                )
                blur_img.append(F.conv2d(input_CBHW, kernel_var, groups=B * C).view((B, C, H, W)))
        return blur_img


class SRMDPreprocess(object):
    def __init__(self, scale, batch_size, sigma, generate_kernel=True):
        self.sig = sigma
        self.b_size = batch_size
        self.scale = scale
        self.gen = generate_kernel

    def __call__(self, hr_tensor, kernel=True):
        # hr_tensor is tensor, not cuda tensor
        hr_var = hr_tensor
        if self.gen:
            kernels, sigs, sizes = batch_random_size_kernel(self.sig, batch_size=self.b_size)
            label_sigma = [(i - 2) for i in sigs]
            label_sigma = torch.from_numpy(np.array(label_sigma))
            # print(sigs, label_sigma)
            label_size = [(i - 11) // 6 for i in sizes]
            label_size = torch.from_numpy(np.array(label_size))
            # print(sizes, label_size)
        else:
            kernels, sizes = batch_stable_kernel(self.sig)

        hr_blur_img = BatchBlur(sizes)(hr_var, kernels)

        hr_blur_img = torch.cat(hr_blur_img)
        # Down sample
        if self.scale != 1:
            lr_blured_t = b_Bicubic(hr_blur_img, self.scale)
        else:
            lr_blured_t = hr_blur_img



        return (lr_blured_t, label_sigma, label_size) if self.gen else lr_blured_t