import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
from .module_util import imresize


def batch_stable_kernel(label_sigma, batch=1, tensor=True, argmax=True):
    kernels = []
    sizes = []
    for sig in (label_sigma):
        if argmax:
            sig = torch.argmax(sig).item() + 2
        if sig == 2:
            l = 11
        elif sig == 3:
            #l = 17
            l = 17
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
    # print('inner:', sig, sizes)
    return kernels, sizes


def b_Bicubic(variable, scale):
    B, C, H, W = variable.size()
    H_new = int(H / scale)
    W_new = int(W / scale)
    tensor_v = variable.view((B, C, H, W))
    re_tensor = imresize(tensor_v, 1 / scale)
    return re_tensor


class BatchBlur(object):
    def __init__(self, l):
        self.l = l
        #print(self.l)
        # self.pad = nn.ZeroPad2d(l // 2)

    def __call__(self, input, kernels):
        blur_img = []
        for i in range(len(self.l)):
            if self.l[i] % 2 == 1:
                self.pad = (self.l[i] // 2, self.l[i] // 2, self.l[i] // 2, self.l[i] // 2)
            else:
                self.pad = (self.l[i] // 2, self.l[i] // 2 - 1, self.l[i] // 2, self.l[i] // 2 - 1)

            B, C, H, W = input[i:i+1, :, :, :].size()
            # print('hr:', input.shape)
            pad = F.pad(input[i:i+1, :, :, :], self.pad, mode='reflect')
            H_p, W_p = pad.size()[-2:]
            # print(kernels[i].shape)
            if len(kernels[i].size()) == 2:
                input_CBHW = pad.view((C * B, 1, H_p, W_p))
                kernel_var = kernels[i].contiguous().view((1, 1, self.l[i], self.l[i])).cuda()
                blur_img.append(F.conv2d(input_CBHW, kernel_var, padding=0).view((B, C, H, W)))
            else:
                input_CBHW = pad.view((1, C * B, H_p, W_p))
                kernel_var = (
                    kernels[i].contiguous()
                    .view((B, 1, self.l[i], self.l[i]))
                    .repeat(1, C, 1, 1)
                    .view((B * C, 1, self.l[i], self.l[i]))
                ).cuda()
                blur_img.append(F.conv2d(input_CBHW, kernel_var, groups=B * C).view((B, C, H, W)))
        return blur_img


class SRMDPreprocess(object):
    def __init__(self, scale, batch_size, sigma, argmax=True):

        self.sig = sigma
        self.b_size = batch_size
        self.scale = scale
        self.arg = argmax

    def __call__(self, hr_tensor, kernel=True):
        # hr_tensor is tensor, not cuda tensor
        hr_var = hr_tensor
        kernels, sizes = batch_stable_kernel(label_sigma=self.sig, argmax=self.arg)

        hr_blur_img = BatchBlur(sizes)(hr_var, kernels)

        hr_blur_img = torch.cat(hr_blur_img)
        # Down sample
        if self.scale != 1:
            lr_blured_t = b_Bicubic(hr_blur_img, self.scale)
        else:
            lr_blured_t = hr_blur_img

        return lr_blured_t


if __name__ == "__main__":
    Batch_Size = 8
    '''out = random_batch_kernel()
    print(len(out[0]))
    print(out[1])
    print(out[2])'''
    '''x = [choice(np.arange(11, 22, 2)) for i in range(Batch_Size)]
    print(x)
    kernels = random_batch_kernel(x)
    img = plt.imread('./butterfly.png')
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    img = img.repeat(Batch_Size, 1, 1, 1)

    perop = SRMDPreprocess(4, x)
    out, label = perop(img)
    print(out.shape)
    print(label)'''
    img = plt.imread('./butterfly.png')
    img1 = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    img1 = img1.repeat(Batch_Size, 1, 1, 1)
    print(img1.shape)
    perop = SRMDPreprocess(1, Batch_Size, random.choice([2, 3, 4]))
    blur_img1, label1, label2 = perop(img1)
    print(label1)
    print(label2)

    '''plt.imshow(blur_img1.permute(0, 2, 3, 1).squeeze(0))
    plt.pause(1)

    img2 = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    perop = SRMDPreprocess(1, [7])
    blur_img2, _ = perop(img2)

    plt.imshow(blur_img2.permute(0, 2, 3, 1).squeeze(0))
    plt.pause(1)

    img_new = blur_img1 - blur_img2
    plt.imshow(img_new.permute(0, 2, 3, 1).squeeze(0), cmap='gray')
    plt.pause(1)
    print(torch.max(img_new))'''