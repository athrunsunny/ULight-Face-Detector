import math
import torch
import torch.nn as nn


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    # default_act = nn.SiLU()  # default activation
    default_act = nn.ReLU(inplace=True)  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvBlock(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.ReLU(inplace=True)  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c1, k, s, autopad(k, p, d), groups=c1, dilation=d, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.act1 = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.conv2 = nn.Conv2d(c1, c2, 1, 1, groups=g, dilation=d, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x

    def forward_fuse(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        return x


class DPConv(nn.Module):
    default_act = nn.ReLU(inplace=True)  # default activation

    def __init__(self, c1, c2, k=1, s=1, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, groups=g, dilation=d, bias=False)
        self.act1 = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.depth_conv = nn.Sequential(
            nn.Conv2d(c2, c2, 3, 2, 1, groups=c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c1, 1)
        )

        self.act2 = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        x = self.act2(self.depth_conv(self.act1(self.conv(x))))
        return x


class RConv(nn.Module):
    default_act = nn.ReLU(inplace=True)  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=1, d=1, act=True):
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c1, k, s, p, groups=c1, dilation=d)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.conv2 = nn.Conv2d(c1, c2, 1, s)

    def forward(self, x):
        x = self.conv2(self.act(self.conv1(x)))
        x = x.permute(0, 2, 3, 1).contiguous()
        return x


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1, s=1):
        super().__init__()
        self.d = dimension
        self.s = s

    def forward(self, x):
        out = torch.cat([o.view(o.size(0), -1, self.s) for o in x], self.d)
        return out


class SimpleConv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=1, g=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x


class Detect(nn.Module):

    def __init__(self, nc=80, imgsz=0, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.imgsz = imgsz

    def forward(self, x):
        return x[1],x[0]
