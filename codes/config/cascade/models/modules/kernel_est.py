import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(
            *[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning


# 定义残差块ResBlock
class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        # 这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        # 将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, ResBlock, num_class1=8, num_class2=5):
        super(ResNet, self).__init__()

        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        self.layer5 = self.make_layer(ResBlock, 512, 2, stride=2)
        self.fc1 = nn.Linear(512, num_class1)

        self.fc2 = nn.Linear(512, num_class2)

    # 这个函数主要是用来，重复同一个残差块
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        #out = self.conv1(x)

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out1 = F.avg_pool2d(out, 4)

        out1 = out1.view(out1.size(0), -1)
        out1 = self.fc1(out1)

        out2 = self.layer5(out)
        out2 = F.avg_pool2d(out2, 2)
        out2 = out2.view(out2.size(0), -1)
        out2 = self.fc2(out2)

        return out1, out2


class Estimate(nn.Module):
    def __init__(self, num_rdb, num_class1, num_class2):
        super(Estimate, self).__init__()
        self.num_rdb = num_rdb

        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.rdbs = nn.ModuleList([RDB(64, 64, 3)])
        for _ in range(self.num_rdb - 1):
            self.rdbs.append(RDB(64, 64, 4))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(64 * num_rdb, 64, kernel_size=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=3 // 2)
        )

        self.res = ResNet(ResBlock, num_class1, num_class2)

    def forward(self, x):
        shallow_feature = self.conv1(x)
        local_features = []
        for i in range(self.num_rdb):
            rdb_out = self.rdbs[i](shallow_feature)
            local_features.append(rdb_out)

        # global residual learning
        gff_out = self.gff(torch.cat(local_features, 1)) + shallow_feature
        return self.res(gff_out)



if __name__ == '__main__':
    # 'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19'
    # Example
    x = torch.randn([1, 3, 32, 32])
    net = Estimate(3, 8, 5)
    out = net(x)
    print(out[0].shape, out[1].shape)

