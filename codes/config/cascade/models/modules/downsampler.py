import torch.nn as nn

from .att_crop import AttentionCropBlock
from .utils import SRMDPreprocess
from .kernel_est import Estimate

class LR_Estimate(nn.Module):
    def __init__(self, scale=4, num_rdb=3):
        super(LR_Estimate, self).__init__()
        self.scale = scale
        self.est = Estimate(num_rdb, 3, 3)
        self.att_crop = AttentionCropBlock(48)

    def forward(self, lr, sr, sigma=[], need_est=True):

        B = sr.shape[0]
        # kernel estimate
        if need_est:
            crop = self.att_crop(lr)
            sigma, size = self.est(crop)
            # downsample sr
            out = SRMDPreprocess(self.scale, B, sigma)(sr)
            return out, sigma, size
        else:
            assert sigma is not None
            out = SRMDPreprocess(self.scale, B, sigma)(sr)
            return out