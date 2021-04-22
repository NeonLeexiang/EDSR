"""
    date:       2021/4/20 4:42 下午
    written by: neonleexiang
"""
import math
import torch
import torch.nn as nn


'''
    原因：为了进行数据特征标准化，即像机器学习中的特征预处理那样对输入特征向量各维去均值再除以标准差，
    但由于自然图像各点像素值的范围都在0-255之间，方差大致一样，
    只要做去均值（减去整个图像数据集的均值或各通道关于图像数据集的均值）处理即可
    
    相当于做一次图像预处理
'''


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=(1, 1))
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std

        for p in self.parameters():
            p.requires_grad = False


'''
    for bias:
        CLASS torch.nn.Conv2d(in_channels, out_channels, 
                              kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
                              
        padding='valid' is the same as no padding. 
        padding='same' pads the input so the output has the shape as the input. 
        However, this mode doesn’t support any stride values other than 1.
        
        
        bias (bool, optional) – If True, adds a learnable bias to the output. Default: True
        
        
'''


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
        '''
            &是按位逻辑运算符，比如5 & 6，5和6转换为二进制是101和110，此时101 & 110=100，100转换为十进制是4，所以5 & 6=4
            
            
            如果一个数是2^n，说明这个二进制里面只有一个1。除了1.
            a  = (10000)b
            a-1 = (01111)b
            a&(a-1) = 0。
            如果一个数不是2^n，
            说明它的二进制里含有多一个1。            
            a = (1xxx100)b            
            a-1=(1xxx011)b         
            那么 a&(a-1)就是 (1xxx000)b，            
            而不会为0。
            
            所以可以用这种方法判断一个数是不2^n。
            
        '''

        '''
            一：与运算符（&）
            运算规则：
            0&0=0；0&1=0；1&0=0；1&1=1           
            即：两个同时为1，结果为1，否则为0
        '''

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
    def __init__(self, n_channels=3, n_resblocks=32, n_feats=256, scale=4, res_scale=1, rgb_range=1):
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







