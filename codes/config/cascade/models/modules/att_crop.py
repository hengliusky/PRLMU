import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class AttentionCropBlock(nn.Module):
    def __init__(self, crop_size):
        super(AttentionCropBlock, self).__init__()
        self.crop_size = crop_size
        self.SA = SpatialAttention()

        self.feat_ext = nn.Sequential(
            conv3x3(3, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv3x3(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv3x3(64, 64),
            nn.BatchNorm2d(64)
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        feat = self.feat_ext(x)
        sa = self.SA(feat)
        row_id, col_id = self.get_highest_area(sa)
        out = torch.zeros([B, C, self.crop_size, self.crop_size]).cuda()
        for i in range(x.shape[0]):
            out[i, :, :, :] = x[i, :, row_id[i]:row_id[i]+self.crop_size, col_id[i]:col_id[i]+self.crop_size]
        return out

    def get_highest_area(self, arr):
        assert len(arr.shape) == 4, arr.shape[2]>self.crop_size and arr.shape[3]>self.crop_size
        row_id, col_id = [], []
        for i in range(arr.shape[0]):
            max_sum = float("-inf")
            row_idx, col_idx = 0, 0
            for row in range(0, arr.shape[2] - self.crop_size, 1):
                for col in range(0, arr.shape[3] - self.crop_size, 1):
                    curr_sum = torch.sum(arr[i, :, row:row + self.crop_size, col:col + self.crop_size])
                    if curr_sum < max_sum:
                        # print(curr_sum)
                        # print(row, col)
                        row_idx, col_idx = row, col
                        max_sum = curr_sum
            row_id.append(row_idx)
            col_id.append(col_idx)
        return row_id, col_id



if __name__ == "__main__":
    x = torch.randn([1, 3, 48, 48])
    net = AttentionCropBlock(32)
    y = net(x)
    print(y.shape)

