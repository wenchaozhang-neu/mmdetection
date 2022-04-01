import warnings
import torch

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS
class Cross_conv(nn.Module):
    def __init__(self):
        super(Cross_conv,self).__init__()
        self.conv2d_1_1_1 = nn.Conv2d(256, 32, kernel_size=1,padding=0)
        self.conv2d_1_3 = nn.Conv2d(256, 32, kernel_size=1,padding=0)
        self.conv2d_1_5 = nn.Conv2d(256, 32, kernel_size=1,padding=0)
        self.conv2d_1_7 = nn.Conv2d(256, 32, kernel_size=1,padding=0)
        self.conv2d_1_0 = nn.Conv2d(256, 32, kernel_size=(1,1), padding=(0,0))
        self.conv2d_1_1 = nn.Conv2d(256, 32, kernel_size=(1,1), padding=(0,0))
        self.conv2d_3_0 = nn.Conv2d(256, 32, kernel_size=(1,3), padding=(0,int(3/2)))
        self.conv2d_3_1 = nn.Conv2d(256, 32, kernel_size=(3,1), padding=(int(3/2),0))
        self.conv2d_5_0 = nn.Conv2d(256, 32, kernel_size=(1,5), padding=(0,int(5/2)))
        self.conv2d_5_1 = nn.Conv2d(256, 32, kernel_size=(5,1), padding=(int(5/2),0))
        self.conv2d_7_0 = nn.Conv2d(256, 32, kernel_size=(1,7), padding=(0,int(7/2)))
        self.conv2d_7_1 = nn.Conv2d(256, 32, kernel_size=(7,1), padding=(int(7/2),0))
        self.bn0 = nn.GroupNorm(4, 32)
        self.bn1 = nn.GroupNorm(4, 32)
        self.bn2 = nn.GroupNorm(4, 32)
        self.bn3 = nn.GroupNorm(4, 32)
        self.bn4 = nn.GroupNorm(4, 32)
        self.bn5 = nn.GroupNorm(4, 32)
        self.bn6 = nn.GroupNorm(4, 32)
        self.bn7 = nn.GroupNorm(4, 32)
        self.bn8 = nn.GroupNorm(4, 32)
        self.bn9 = nn.GroupNorm(4, 32)
        self.bn10 = nn.GroupNorm(4, 32)
        self.bn11 = nn.GroupNorm(4, 32)

    def forward(self, x):
        x_1_3 = self.bn0(self.conv2d_1_1_1(x))
        x_1_0 = self.bn1(self.conv2d_1_0(x))
        x_1_1 = self.bn2(self.conv2d_1_1(x))
        x1 = torch.cat((x_1_3, (x_1_0+x_1_1)), 1)
        x_3_3 = self.bn3(self.conv2d_1_3(x))
        x_3_0 = self.bn4(self.conv2d_3_0(x))
        x_3_1 = self.bn5(self.conv2d_3_1(x))
        x3 = torch.cat((x_3_3, (x_3_0+x_3_1)), 1)
        x_5_3 = self.bn6(self.conv2d_1_5(x))
        x_5_0 = self.bn7(self.conv2d_5_0(x))
        x_5_1 = self.bn8(self.conv2d_5_1(x))
        x5 = torch.cat((x_5_3, (x_5_0+x_5_1)), 1)
        x_7_3 = self.bn9(self.conv2d_1_7(x))
        x_7_0 = self.bn10(self.conv2d_7_0(x))
        x_7_1 = self.bn11(self.conv2d_7_1(x))
        x7 = torch.cat((x_7_3, (x_7_0+x_7_1)), 1)
        x = torch.cat((x1, x3, x5, x7), 1)
        return x
class ECA(nn.Module):
    """docstring for Eca_block"""
    def __init__(self):
        super(ECA, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=5, padding=int(5/2), bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, gamma=2, b=1):
        #N, C, H, W = x.size()
        y = self.avgpool(x)
        y = self.conv1d(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return x*y.expand_as(x)

class Group_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Group_conv, self).__init__()
        self.conv2d_1 = nn.Conv2d(in_channels, int(out_channels/2), kernel_size=1, padding=0, groups=2)
        self.conv2d_3 = nn.Conv2d(in_channels, int(out_channels/2), kernel_size=3, padding=int(3/2), groups=16)
        self.conv2d_5 = nn.Conv2d(in_channels, int(out_channels/2), kernel_size=3, padding=int(3/2), groups=16)
        self.eca = ECA()
        self.bn0 = nn.GroupNorm(16, int(out_channels/2))
        self.bn1 = nn.GroupNorm(16, int(out_channels/2))
        self.bn2 = nn.GroupNorm(16, int(out_channels/2))
        #self.bn3 = nn.GroupNorm(32, 256)

    def forward(self, x):

        x1 = self.bn0(self.conv2d_1(x))
        x3 = self.bn1(self.conv2d_3(x))
        x5 = self.bn2(self.conv2d_5(x))
        x = torch.cat((x1,(x3 + x5)),1)
        x = self.eca(x)
        return x

@NECKS.register_module()
class FPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(FPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # TODO: deprecate `extra_convs_on_inputs`
                warnings.simplefilter('once')
                warnings.warn(
                    '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
                    'Please use "add_extra_convs"', DeprecationWarning)
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            #l_conv = ConvModule(
            #    in_channels[i],
            #    out_channels,
                #     1,
           #     conv_cfg=conv_cfg,
           #     norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
           #     act_cfg=act_cfg,
           #     inplace=False)
            fpn_conv = Cross_conv()
            l_conv = Group_conv(in_channels[i], out_channels) 
            #fpn_conv = ConvModule(
           #     out_channels,
           #     out_channels,
              #       3,
           #     padding=1,
          #      conv_cfg=conv_cfg,
          #      norm_cfg=norm_cfg,
          #      act_cfg=act_cfg,
          #      inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                          3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        torch.cuda.synchronize()
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        torch.cuda.synchronize()
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
