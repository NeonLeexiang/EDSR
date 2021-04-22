"""
    date:       2021/4/20 4:42 下午
    written by: neonleexiang
"""
import math
import torch
import torch.nn as nn


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=(1, 1))
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std

        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, padding, bias=False, bn=False, act=nn.ReLU(inplace=True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []

        for i in range(2):
            m.append(nn.Conv2d(in_channels=n_feats, out_channels=n_feats,
                               kernel_size=kernel_size, padding=padding, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats, bn=False, act=False, bias=False):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(in_channels=n_feats, out_channels=4 * n_feats,
                                   kernel_size=(3, 3), padding=(1, 1), bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(nn.Conv2d(in_channels=n_feats, out_channels=9 * n_feats,
                               kernel_size=(3, 3), padding=(1, 1), bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))

        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class EDSR(nn.Module):
    def __init__(self, n_channels=3, n_resblocks=32, n_feats=256, scale=4, res_scale=1, rgb_range=255):
        super(EDSR, self).__init__()

        self.n_channels = n_channels
        self.n_resblocks = n_resblocks
        self.n_feats = n_feats
        self.scale = scale
        self.res_scale = res_scale
        self.rgb_range = rgb_range

        self.kernel_size = (3, 3)
        self.padding = (1, 1)
        self.act = nn.ReLU(True)

        self.sub_mean = MeanShift(self.rgb_range)
        self.add_mean = MeanShift(self.rgb_range, sign=1)

        net_head = [nn.Conv2d(self.n_channels, self.n_feats, kernel_size=self.kernel_size, padding=self.padding)]

        net_body = [
            ResBlock(
                n_feats=self.n_feats, kernel_size=self.kernel_size, padding=self.padding,
                act=self.act, res_scale=self.res_scale
            ) for _ in range(self.n_resblocks)
        ]

        net_body.append(nn.Conv2d(in_channels=self.n_feats, out_channels=self.n_feats,
                                  kernel_size=self.kernel_size, padding=self.padding))

        net_tail = [
            Upsampler(self.scale, self.n_feats, act=False),
            nn.Conv2d(in_channels=self.n_feats, out_channels=self.n_channels,
                      kernel_size=self.kernel_size, padding=self.padding)
        ]

        self.net_head = nn.Sequential(*net_head)
        self.net_body = nn.Sequential(*net_body)
        self.net_tail = nn.Sequential(*net_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.net_head(x)

        res = self.net_body(x)
        res = torch.add(x, res)

        x = self.net_tail(res)
        x = self.add_mean(x)

        return x







