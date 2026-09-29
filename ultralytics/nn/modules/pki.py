import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvFFN(nn.Module):
    def __init__(self, in_ch, out_ch, scale=1.0, kernel_size=3, drop=0., add_identity=True):
        super().__init__()
        hidden = int(in_ch * scale)
        self.conv1 = Conv(in_ch, hidden, 1)
        self.conv2 = Conv(hidden, hidden, kernel_size, g=hidden)
        self.conv3 = Conv(hidden, out_ch, 1, act=False)
        self.drop = nn.Dropout(drop) if drop else nn.Identity()
        self.add_identity = add_identity and in_ch == out_ch

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.drop(out)
        out = self.conv3(out)
        if self.add_identity:
            out += x
        return out


class CAA(nn.Module):
    def __init__(self, channels, kernel=7):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_h = nn.Conv2d(channels, channels, (kernel, 1), 1, (kernel // 2, 0), groups=channels)
        self.conv_w = nn.Conv2d(channels, channels, (1, kernel), 1, (0, kernel // 2), groups=channels)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.conv_h(self.pool_h(x).expand(-1, -1, h, w))
        x_w = self.conv_w(self.pool_w(x).expand(-1, -1, h, w))
        return identity * torch.sigmoid(x_h + x_w)


class InceptionBottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_sizes=(3, 5, 7),
                 dilations=(1, 1, 1), expansion=1.0, add_identity=True,
                 with_caa=True, caa_kernel_size=7):
        super().__init__()
        hidden = int(out_ch * expansion)
        self.conv1 = Conv(in_ch, hidden, 1)
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(hidden, hidden, k, 1, (k - 1) // 2 * d, groups=hidden, bias=False)
            for k, d in zip(kernel_sizes, dilations)
        ])
        self.conv2 = Conv(hidden * len(kernel_sizes), out_ch, 1, act=False)
        self.add_identity = add_identity and in_ch == out_ch
        self.caa = CAA(out_ch, caa_kernel_size) if with_caa else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        outs = [dw(out) for dw in self.dw_convs]
        out = self.conv2(torch.cat(outs, 1))
        out = self.caa(out)
        if self.add_identity:
            out += x
        return out


class PKIBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_sizes=(3, 5, 7),
                 dilations=(1, 1, 1), expansion=0.25, ffn_scale=1.0, dropout_rate=0.,
                 drop_path_rate=0., layer_scale=1.0, add_identity=True, with_caa=True,
                 caa_kernel_size=7):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden = int(in_channels * expansion)

        #print(f"PKIBlock init: in_channels={in_channels}, hidden={hidden}, out_channels={out_channels}")

        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(hidden)

        self.block = InceptionBottleneck(
            in_channels, hidden, kernel_sizes, dilations, expansion=1.0,
            add_identity=True, with_caa=with_caa, caa_kernel_size=caa_kernel_size
        )
        self.ffn = ConvFFN(
            hidden, out_channels, ffn_scale, kernel_size=3, drop=dropout_rate, add_identity=False
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.layer_scale = layer_scale
        if layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(hidden))
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(out_channels))

        self.add_identity = add_identity and in_channels == out_channels

        if in_channels != hidden:
            self.downsample1 = nn.Conv2d(in_channels, hidden, kernel_size=1)
        else:
            self.downsample1 = nn.Identity()

        if hidden != out_channels:
            self.downsample2 = nn.Conv2d(hidden, out_channels, kernel_size=1)
        else:
            self.downsample2 = nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.block(x)

        if self.layer_scale:
            x = self.downsample1(shortcut) + self.drop_path(self.gamma1.view(1, -1, 1, 1) * x)
        else:
            x = self.downsample1(shortcut) + self.drop_path(x)

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)

        if self.layer_scale:
            x = self.downsample2(shortcut) + self.drop_path(self.gamma2.view(1, -1, 1, 1) * x)
        else:
            x = self.downsample2(shortcut) + self.drop_path(x)
        return x


class PKIBlock_YOLO(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.25, k=(3, 5, 7), with_caa=True, caa_ks=7):
        super().__init__()
        #for i in range(n):
          #  print(f"PKIBlock_YOLO: block {i+1}/{n}, c1={c1}, c2={c2}, expansion={e}, kernels={k}")
        self.m = nn.Sequential(*[
            PKIBlock(c1, c2, expansion=e, kernel_sizes=k, with_caa=with_caa, caa_kernel_size=caa_ks)
            for _ in range(n)
        ])

    def forward(self, x):
        return self.m(x)


__all__ = ['PKIBlock', 'PKIBlock_YOLO']
