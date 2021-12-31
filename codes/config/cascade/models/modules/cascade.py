import torch
import torch.nn as nn
import math

from .downsampler import LR_Estimate


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

# self-made attention
class ICALayer(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.maxpool(x)
        avg_out = self.avgpool(x)
        max_out = max_out.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        avg_out = avg_out.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        out = self.conv(max_out) + self.conv(avg_out)  # bs,1,c
        out = self.sigmoid(out)  # bs,1,c
        out =out .permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
        return x*out.expand_as(out)


## Residual Channel Attention Block (RCAB)
class ICAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ICAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(ICALayer())
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Residual Group (RG)
class ICAU(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ICAU, self).__init__()
        modules_body = [
            ICAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Residual Channel Attention Network (RCAN)
class Upsampling(nn.Module):
    def __init__(self, add_nc=3):
        super(Upsampling, self).__init__()

        n_resgroups = 1
        n_resblocks = 5
        n_feats = 64
        kernel_size = 3
        reduction = 2
        scale = 4
        act = nn.ReLU(True)

        # define head module
        modules_head = [self.conv(3, n_feats, kernel_size)]
        self.ma = self.conv(add_nc, n_feats, kernel_size)
        # define body module
        modules_body = [
            ICAU(
                self.conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]
        modules_body.append(self.conv(n_feats, n_feats, kernel_size))
        self.up = Upsampler(self.conv, scale, n_feats, act=False)

        para = [nn.Conv2d(n_feats, n_feats, 1, stride=1, padding=0),
                       nn.Conv2d(n_feats, n_feats, 1, stride=1, padding=0),
                        nn.Softmax(dim=1)]
        self.para = nn.Sequential(*para)
        self.conv1x1 = nn.Conv2d(n_feats * 2, n_feats, 1, stride=1, padding=0)
        refine = [
            ICAU(
                self.conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]
        refine.append(self.conv(n_feats, 3, kernel_size))

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.refine = nn.Sequential(*refine)


    def forward(self, x, extra, head=0):
        x = self.head(x)
        ex = self.ma(extra)
        res = self.body(x)
        res += x
        up = self.up(res)
        if head == 0:
            param = self.para(up)
            addition = torch.mul(param, ex)
            f = torch.cat([up, addition], dim=1)
            f = self.conv1x1(f)
        else:
            f = up
        f = self.refine(f)

        return f

    def conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

class PRLMU(nn.Module):
    def __init__(self, in_nc=3):
        super(PRLMU, self).__init__()
        self.Restorer1 = Upsampling(add_nc=3)
        self.Restorer2 = Upsampling(add_nc=3)
        # self.Restorer3 = Upsampling(add_nc=6)
        # self.Restorer4 = Upsampling(add_nc=9)
        self.Restorers = [self.Restorer2]#, self.Restorer3] #self.Restorer3, self.Restorer4]
        self.Downsampler = LR_Estimate()

    def forward(self, input):
        res = []
        srs = []
        extra = torch.rand([1, 3, 256, 256]).cuda()
        sr = self.Restorer1(input, extra, head=1)
        srs.append(sr)
        M = sr
        out = sr
        # size, sigma = torch.randn([1, 3]).cuda(), torch.randn([1, 3]).cuda()
        # lr = self.Downsampler(input, out, sigma, need_est=False)
        lr, sigma, size = self.Downsampler(input, out)
        # print(sigma.shape)
        # resdiual update
        x = input - lr
        res.append(x)
        for net in self.Restorers:
            sr = net(x, M)
            M = torch.cat([M, sr], dim=1)
            out = torch.add(out, sr)
            srs.append(out)
            lr = self.Downsampler(input, out, sigma, need_est=False)
            # residual update
            x = input - lr
            res.append(x)
        return out, res, sigma, size
