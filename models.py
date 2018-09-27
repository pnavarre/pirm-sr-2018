#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pablo Navarrete Michelini
"""
import torch
import numpy as np
from torch import nn
from collections import OrderedDict
from layers import Cutborder, ClassicUpscale, Activ, FilterBlock


class G_MGBP(nn.Module):
    def __init__(self,
                 model_id,
                 name='G-MGBP',
                 str_tab='',
                 vlevel=0):
        self.model_id = model_id
        self.name = name
        self.str_tab = str_tab
        self.vlevel = vlevel
        super().__init__()

        if self.model_id.endswith('_v3_ms'):
            parse = self.model_id.split('_')
            self.factor = [2, 2]
            self._levels = 1
            self.mu = 2

            self.input_channels = int([s for s in parse if s.startswith('CH')][0][2:])
            FE = int([s for s in parse if s.startswith('FE')][0][2:])
            if len([s for s in parse if s.startswith('GR')]) > 0:
                GR = int([s for s in parse if s.startswith('GR')][0][2:])
            else:
                GR = 16
            TR = (len([s for s in parse if s.startswith('TR0')]) == 0)
            bnorm = None
            leak =  float([s for s in parse if s.startswith('LEAK')][0][4:])
            if [s for s in parse if s.startswith('BN')][0] == 'BN1':
                bnorm = nn.BatchNorm2d

            self.net = {}

            if len([s for s in parse if s.startswith('BIC')]) > 0:
                self.net['ClassicUpscale'] = ClassicUpscale(
                    self.input_channels,
                    (2, 2), mode='bicubic', train=True
                )
            else:
                assert False

            an_parse = [s for s in parse if s.startswith('Analysis')][0].split('#')[1:]
            an_k = int([s for s in an_parse if s.startswith('K')][0][1:])
            assert (an_k-1) % 2 == 0
            an_d = int([s for s in an_parse if s.startswith('D')][0][1:])
            an_nlayers = int([s for s in an_parse if s.startswith('L')][0][1:])
            assert an_nlayers > 0
            self.net['Analysis'] = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(
                    self.input_channels, FE, an_k, padding=(an_k-1)//2, dilation=an_d, bias=False)
                 ),
            ]))
            self.net['Analysis'].add_module('block', FilterBlock(
                an_nlayers-1, FE, bnorm, leak, transpose=False,
                kernel_size=an_k, padding=(an_k-1)//2, dilation=an_d,
                start_label=2,
                growth_rate=GR
            ))

            up_parse = [s for s in parse if s.startswith('Upscaling')][0].split('#')[1:]
            up_k = int([s for s in up_parse if s.startswith('K')][0][1:])
            assert (up_k-1) % 2 == 0
            up_d = int([s for s in up_parse if s.startswith('D')][0][1:])
            up_m = int([s for s in up_parse if s.startswith('M')][0][1:])
            up_nlayers = int([s for s in up_parse if s.startswith('L')][0][1:])
            assert up_nlayers >= 0
            assert up_m <= up_nlayers
            up_nlayers1 = up_m
            up_nlayers2 = up_nlayers - up_nlayers1
            self.net['Upscaling'] = nn.Sequential()
            self.net['Upscaling'].add_module('pre_up', FilterBlock(
                up_nlayers1, 2*FE+1, bnorm, leak, transpose=False,
                kernel_size=up_k, padding=(up_k-1)//2, dilation=up_d,
                growth_rate=GR
            ))
            self.net['Upscaling'].add_module(
                'act_up', Activ(2*FE+1, bnorm, leak)
            )
            self.net['Upscaling'].add_module(
                'conv_up', nn.ConvTranspose2d(
                    2*FE+1, FE, up_k, stride=(2, 2),
                    padding=(up_k-1)//2, output_padding=(up_k-1)//2,
                    dilation=1, bias=False
                )
            )
            if up_nlayers2 > 1:
                self.net['Upscaling'].add_module('post_up', FilterBlock(
                    up_nlayers2, FE, bnorm, leak, transpose=False,
                    kernel_size=up_k, padding=(up_k-1)//2, dilation=up_d,
                    start_label=(up_nlayers1+2),
                    growth_rate=GR
                ))

            down_parse = [s for s in parse if s.startswith('Downscaling')][0].split('#')[1:]
            down_k = int([s for s in down_parse if s.startswith('K')][0][1:])
            assert (down_k-1) % 2 == 0
            down_d = int([s for s in down_parse if s.startswith('D')][0][1:])
            down_m = int([s for s in down_parse if s.startswith('M')][0][1:])
            down_nlayers = int([s for s in down_parse if s.startswith('L')][0][1:])
            assert down_nlayers > 0
            assert down_m <= down_nlayers
            down_nlayers1 = down_m
            down_nlayers2 = down_nlayers - down_nlayers1
            self.net['Downscaling'] = nn.Sequential()
            if down_nlayers1 > 0:
                self.net['Downscaling'].add_module('pre_down', FilterBlock(
                    down_nlayers1, FE, bnorm, leak, transpose=TR,
                    kernel_size=down_k, padding=(down_k-1)//2, dilation=down_d,
                    growth_rate=GR
                ))
            self.net['Downscaling'].add_module(
                'conv_down', nn.Conv2d(
                    FE, FE, down_k, stride=(2, 2), padding=(down_k-1)//2, dilation=1, bias=False
                )
            )
            self.net['Downscaling'].add_module('post_down', FilterBlock(
                down_nlayers2, FE, bnorm, leak, transpose=TR,
                kernel_size=down_k, padding=(down_k-1)//2, dilation=down_d,
                start_label=(down_nlayers1+1),
                growth_rate=GR
            ))

            syn_parse = [s for s in parse if s.startswith('Synthesis')][0].split('#')[1:]
            syn_k = int([s for s in syn_parse if s.startswith('K')][0][1:])
            assert (syn_k-1) % 2 == 0
            syn_d = int([s for s in syn_parse if s.startswith('D')][0][1:])
            syn_nlayers = int([s for s in syn_parse if s.startswith('L')][0][1:])
            assert syn_nlayers > 0

            self.net['Synthesis'] = nn.Sequential()
            self.net['Synthesis'].add_module('block', FilterBlock(
                syn_nlayers-1, FE, bnorm, leak, transpose=False,
                kernel_size=syn_k, padding=(syn_k-1)//2, dilation=syn_d,
                growth_rate=GR
            ))
            self.net['Synthesis'].add_module(
                'act%d' % syn_nlayers, Activ(FE, bnorm, leak)
            )
            self.net['Synthesis'].add_module(
                'conv%d' % syn_nlayers, nn.Conv2d(
                    FE, self.input_channels, syn_k, padding=(syn_k-1)//2, dilation=syn_d, bias=True
                )
            )
        else:
            assert False
        self.border_in = (0, 0, 0, 0)
        self.border_out = (0, 0, 0, 0)
        self.border_cut = (0, 0, 0, 0)
        self._cached_border_in = {}
        self._cached_border_out = {}
        self._cached_border_cut = {}

        if self.model_id.endswith('_v3_ms'):
            self.border_in = (self.factor[0], ) * 4
            self.border_cut = (self.factor[0]*self.factor[0], ) * 4
            self.reflection = torch.nn.ReflectionPad2d(self.border_in)
        else:
            assert False

        if self.model_id.endswith('_ms'):
            for name, layer in self.net.items():
                self.add_module(name, layer)

        self.set_factor(self.factor)

        self.stat_nparam = 0
        for name, par in self.named_parameters():
            if name.endswith('weight'):
                self.stat_nparam += np.prod(par.shape)
            if name.endswith('bias'):
                self.stat_nparam += np.prod(par.shape)

        if self.vlevel > 0:
            print(self.str_tab, self.name,
                  '- Model', self.model_id)
            print(self.str_tab, self.name,
                  '- Factor set to %dx%d' % (self.factor[0], self.factor[1]))
            print(self)
            print(self.str_tab, self.name,
                  '- border_out', self.border_out)
            print(self.str_tab, self.name,
                  '- border_in', self.border_in)
            print(self.str_tab, self.name,
                  '- border_cut', self.border_cut)
            print(self.str_tab, self.name,
                  '- # weight/bias: {:_}'.format(self.stat_nparam))

    def _v3_RecBP(self, hr, level, lowest=0):
        new_hr = hr
        if level > lowest:
            for k in range(self.mu):
                lr = self.net['Downscaling'](new_hr)
                update = self._v3_RecBP(lr, level-1, lowest)
                new_hr = new_hr + self.net['Upscaling'](
                    torch.cat([
                        self._v3_res[level-1],
                        update,
                        self._v3_noise[level-1]
                    ], 1)
                )
            return new_hr
        else:
            return hr

    def forward(self, x, noise_amp=None, pad=False, max_levels=3):
        x_pad = x
        if pad:
            x_pad = self.reflection(x)

        if self.model_id.endswith('_v3_ms'):
            assert noise_amp is not None
            out = x_pad
            self._v3_res = (self._levels+1)*[None]
            self._v3_noise = (self._levels+1)*[None]

            self._v3_res[0] = self.net['Analysis'](x_pad)
            self._v3_noise[0] = noise_amp * torch.randn(
                [self._v3_res[0].shape[0], 1,
                 self._v3_res[0].shape[2],
                 self._v3_res[0].shape[3]]
            ).to(self._v3_res[0])
            out = out + self.net['Synthesis'](self._v3_res[0])

            res = self._v3_res[0]
            for lr_level in range(self._levels):
                out = self.net['ClassicUpscale'](out, padding=pad)
                downres = self.net['Downscaling'](self.net['Analysis'](out))
                res = self.net['Upscaling'](torch.cat([
                    res,
                    downres,
                    self._v3_noise[lr_level]
                ], 1))
                res = self._v3_RecBP(res, lr_level+1, lowest=max(0, (lr_level+1)-max_levels))
                self._v3_res[lr_level+1] = res
                self._v3_noise[lr_level+1] = noise_amp * torch.randn(
                    [res.shape[0], 1, res.shape[2], res.shape[3]]
                ).to(res)
                out = out + self.net['Synthesis'](res)

        else:
            out = self.net(x_pad)

        if pad:
            return Cutborder(out, self.border_cut)
        return out

    def set_mu(self, mu):
        if self.model_id.endswith('_v3_ms'):
            self.mu = mu
            self.set_factor(force=True)

    def set_factor(self, factor=None, force=False):
        if factor is None:
            factor = self.factor
        s = '%dx%d' % (factor[0], factor[1])
        if s in self._cached_border_in and not force:
            self._levels = int(np.log2(factor[0]))
            self.factor = factor
            self.border_in = self._cached_border_in[s]
            self.border_out = self._cached_border_out[s]
            self.border_cut = self._cached_border_cut[s]
            self.reflection = nn.ReflectionPad2d(self.border_in)
        elif self.model_id.endswith('_v3_ms'):
            assert len(factor) == 2
            assert np.log2(factor[0]) == np.floor(np.log2(factor[0]))
            assert np.log2(factor[1]) == np.floor(np.log2(factor[1]))
            if self.vlevel > 0:
                print(self.str_tab, self.name,
                      '- Setting factor %dx%d' % (factor[0], factor[1]))
            assert self.factor[0] == self.factor[1]
            assert factor[0] == factor[1]
            self._levels = int(np.log2(factor[0]))
            self.factor = factor

            self.border_in = (self.factor[0], ) * 4
            self.border_cut = (self.factor[0]*self.factor[0], ) * 4
            self.reflection = torch.nn.ReflectionPad2d(self.border_in)
            self.border_out = (0, 0, 0, 0)
            self._cached_border_in[s] = self.border_in
            self._cached_border_out[s] = self.border_out
            self._cached_border_cut[s] = self.border_cut
        else:
            assert len(factor) == 2
            assert np.log2(factor[0]) == np.floor(np.log2(factor[0]))
            assert np.log2(factor[1]) == np.floor(np.log2(factor[1]))
            if self.vlevel > 0:
                print(self.str_tab, self.name,
                      '- Setting factor %dx%d' % (factor[0], factor[1]))
            if self.model_id.endswith('_ms'):
                assert self.factor[0] == self.factor[1]
                assert factor[0] == factor[1]
                self._levels = int(np.log2(factor[0]))
                self.factor = factor

                self.border_in = (0, 0, 0, 0)
                self.border_out = (0, 0, 0, 0)
                self.border_cut = (0, 0, 0, 0)

                probe_shape = (1, self.input_channels, 4, 4)
                probe = torch.zeros(probe_shape, requires_grad=False)

                out_size = self.forward(probe, pad=False).shape
                pad_v = probe.shape[2]*self.factor[0]-out_size[2]
                if pad_v > 0:
                    pad_v += (self.factor[0]*2) - (pad_v % (self.factor[0]*2))
                pad_h = probe.shape[3]*self.factor[1]-out_size[3]
                if pad_h > 0:
                    pad_h += (self.factor[1]*2) - (pad_h % (self.factor[1]*2))
                self.border_out = (
                    pad_h//2, pad_h-pad_h//2,
                    pad_v//2, pad_v-pad_v//2
                )
                assert np.all(np.asarray(self.border_out) >= 0)

                assert pad_v % (2*self.factor[0]) == 0
                assert pad_h % (2*self.factor[1]) == 0
                pad_v = pad_v // self.factor[0]
                pad_h = pad_h // self.factor[1]
                self.border_in = (
                    pad_h//2, pad_h-pad_h//2,
                    pad_v//2, pad_v-pad_v//2
                )
                assert np.all(np.asarray(self.border_in) >= 0)

                self.reflection = nn.ReflectionPad2d(self.border_in)

                outwithpad_size = self.forward(probe, pad=True).shape
                pad_v = max(outwithpad_size[2] - probe.shape[2]*self.factor[0], 0)
                pad_h = max(outwithpad_size[3] - probe.shape[3]*self.factor[1], 0)
                self.border_cut = (
                    pad_h//2, pad_h-pad_h//2,
                    pad_v//2, pad_v-pad_v//2
                )
                assert np.all(np.asarray(self.border_cut) >= 0)

                if self.vlevel > 0:
                    print(self.str_tab, self.name,
                          '- probe_shape', probe.data.shape)
                    print(self.str_tab, self.name,
                          '- out_size', out_size)
                    print(self.str_tab, self.name,
                          '- outwithpad_size', outwithpad_size)
                    print(self.str_tab, self.name,
                          '- border_out', self.border_out)
                    print(self.str_tab, self.name,
                          '- border_in', self.border_in)
                    print(self.str_tab, self.name,
                          '- border_cut', self.border_cut)
            else:
                assert factor[0] == self.factor[0]
                assert factor[1] == self.factor[1]
            self._cached_border_in[s] = self.border_in
            self._cached_border_out[s] = self.border_out
            self._cached_border_cut[s] = self.border_cut

    def load_state_dict(self, state_dict, strict=False):
        from torch.nn.parameter import Parameter
        own_state = self.state_dict()
        for name, param in state_dict.items():
            name = name.replace('act.1', 'act_1')
            name = name.replace('act.2', 'act_2')
            name = name.replace('conv.1', 'conv_1')
            name = name.replace('conv.2', 'conv_2')
            if name in own_state:
                if isinstance(param, Parameter):
                    param = param.data
                try:
                    if name.split('.')[-1] == 'bias':
                        own_state[name][:param.shape[0]].copy_(param)
                    elif name.split('.')[-1] == 'weight':
                        own_state[name][:param.shape[0], :param.shape[1]].copy_(param)
                    else:
                        own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))
            elif strict:
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
        else:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                print('WARNING: missing keys in state_dict: "{}"'.format(missing))

