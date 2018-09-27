#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pablo Navarrete Michelini
"""
import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter


def kernel_cubic(zoom, phase, length=None):
    assert zoom > 0

    lower_bound = np.ceil(-2*zoom-phase)
    higher_bound = np.floor(2*zoom-phase)

    anchor = max(abs(lower_bound), abs(higher_bound))
    index = np.arange(-anchor+1, anchor+1)
    if length is not None:
        assert length >= 2*anchor
        anchor = np.ceil(length/2)
        index = np.arange(-anchor+1, length-anchor+1)

    pos = abs(index+phase) / zoom

    kernel = np.zeros(np.size(pos))
    idx = (pos < 2)
    kernel[idx] = -0.5 * pos[idx]**3 + 2.5 * pos[idx]**2 - 4*pos[idx] + 2
    idx = (pos < 1)
    kernel[idx] = 1.5 * pos[idx]**3 - 2.5 * pos[idx]**2 + 1

    kernel = kernel * zoom / np.sum(kernel)

    return kernel


def Cutborder(input, border, feature=None):
    assert isinstance(border, tuple)
    if border == (0, 0, 0, 0):
        return input
    left, right, top, bottom = border
    assert top >= 0 and bottom >= 0 and \
        left >= 0 and right >= 0
    if np.all(np.asarray(border) == 0) and feature is None:
        return input
    if bottom == 0:
        bottom = None
    else:
        bottom = -bottom
    if right == 0:
        right = None
    else:
        right = -right
    if feature is None:
        return input[:, :, top:bottom, left:right]
    else:
        return input[:, feature:feature+1, top:bottom, left:right]


class Activ(nn.Module):
    def __init__(self, features, norm, leak=0, inplace=True):
        super().__init__()
        self.leak = leak
        self.inplace = inplace
        if norm is None:
            self.norm = Bias(features)
        else:
            self.norm = norm(features)
        if self.leak == 0:
            self.act = nn.ReLU(inplace=self.inplace)
        else:
            self.act = nn.LeakyReLU(
                negative_slope=self.leak,
                inplace=self.inplace
            )

    def forward(self, input):
        return self.act(self.norm(input))

    def __repr__(self):
        return str(self.norm) + ', ' + str(self.act)


class _ClassicScaler(nn.Module):
    def __init__(self, channels, stride, mode='bicubic', param=3, train=False):
        assert len(stride) == 2
        assert isinstance(stride[0], int) and isinstance(stride[1], int)
        self.channels = channels
        self.stride = stride
        self.mode = mode
        self.param = param
        self._train = train
        super().__init__()

        shift = (
            -stride[0]/2 + 0.5,
            -stride[0]/2 + 0.5
        )
        if mode == 'bicubic':
            fh = np.asarray([
                kernel_cubic(self.stride[1], shift[1])
            ]) / self.stride[1]
            fv = np.asarray([
                kernel_cubic(self.stride[0], shift[0])
            ]) / self.stride[0]
        else:
            assert False, 'Mode %s not supported' % mode

        f2d = fh * fv.T
        f = np.zeros([
            self.channels,  1,
            f2d.shape[0]+1,
            f2d.shape[1]+1,
        ])
        for c in range(self.channels):
            f[c, 0, 1:, 1:] = np.asarray(f2d)
        if self._train:
            self.weight = Parameter(torch.FloatTensor(np.asarray(f)))
        else:
            self.register_buffer(
                'weight', Variable(torch.FloatTensor(np.asarray(f)))
            )

    def __repr__(self):
        s = '{name}({channels}, stride={stride}), mode={mode}, ' \
            'param={param}, kernel_size=%dx%dx%dx%d)' % \
            (self.weight.shape[0], self.weight.shape[1],
             self.weight.shape[2], self.weight.shape[3])
        return s.format(name=self.__class__.__name__, **self.__dict__)


class ClassicUpscale(_ClassicScaler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reflection = nn.ReplicationPad2d((1, 1, 1, 1))
        self.zeropad = nn.ZeroPad2d((1, 1, 1, 1))
        self._vL = (2*self.stride[0] + self.weight.shape[2])//2
        self._vR = self._vL - self.stride[0] + 1
        self._hU = (2*self.stride[1] + self.weight.shape[3])//2
        self._hB = self._hU - self.stride[1] + 1
        self._wfactor = self.stride[0] * self.stride[1]

    def forward(self, x, padding=True):
        xpad = self.reflection(x) if padding else self.zeropad(x)
        out = nn.functional.conv_transpose2d(
            input=xpad,
            weight=self._wfactor * self.weight,
            stride=self.stride,
            bias=None,
            padding=0,
            output_padding=0,
            dilation=1,
            groups=x.shape[1]
        )

        return out[:, :, self._vL:-self._vR, self._hU:-self._hB]


class Bias(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.bias = Parameter(torch.zeros(features))

    def forward(self, input):
        return input + self.bias.\
            unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(input)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.features) + ')'


class FilterBlock(nn.Sequential):
    def __init__(self, depth,
                 num_features,
                 bnorm,
                 leak,
                 transpose,
                 growth_rate=16,
                 kernel_size=3,
                 padding=1,
                 dilation=1,
                 start_label=1
                 ):
        super().__init__()
        self.add_module(
            'dense',
            DenseBlock(
                num_layers=depth,
                num_input_features=num_features,
                bneck_factor=4,
                growth_rate=growth_rate,
                drop_rate=0,
                bnorm=bnorm,
                leak=leak,
                transpose=transpose
            )
        )
        self.add_module(
            'transition',
            DenseTransition(
                num_input_features=num_features + depth * growth_rate,
                num_output_features=num_features,
                bnorm=bnorm,
                leak=leak,
                transpose=transpose
            )
        )


class DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bneck_factor, drop_rate, bnorm, leak, transpose):
        super().__init__()
        self.add_module('act_1', Activ(num_input_features, bnorm, leak))
        if transpose:
            self.add_module('conv_1', nn.ConvTranspose2d(num_input_features, bneck_factor *
                            growth_rate, kernel_size=1, stride=1, bias=False))
        else:
            self.add_module('conv_1', nn.Conv2d(num_input_features, bneck_factor *
                            growth_rate, kernel_size=1, stride=1, bias=False))
        self.add_module('act_2', Activ(bneck_factor * growth_rate, bnorm, leak))
        if transpose:
            self.add_module('conv_2', nn.ConvTranspose2d(bneck_factor * growth_rate, growth_rate,
                            kernel_size=3, stride=1, padding=1, bias=False))
        else:
            self.add_module('conv_2', nn.Conv2d(bneck_factor * growth_rate, growth_rate,
                            kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bneck_factor, growth_rate, drop_rate, bnorm, leak, transpose):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bneck_factor, drop_rate, bnorm, leak, transpose)
            self.add_module('denselayer%d' % (i + 1), layer)


class DenseTransition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, bnorm, leak, transpose):
        super().__init__()
        self.add_module('act', Activ(num_input_features, bnorm, leak))
        if transpose:
            self.add_module(
                'conv',
                nn.ConvTranspose2d(num_input_features, num_output_features,
                                   kernel_size=1, stride=1, bias=False)
            )
        else:
            self.add_module(
                'conv',
                nn.Conv2d(num_input_features, num_output_features,
                          kernel_size=1, stride=1, bias=False)
            )
