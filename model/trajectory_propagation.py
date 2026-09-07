import torch
import torch.nn as nn
import torchvision

from model.misc import constant_init
from model.modules.deformconv import ModulatedDeformConv2d


class TrajectoryDeformableAlignment(ModulatedDeformConv2d):
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 3)
        super().__init__(*args, **kwargs)
        input_channels = 2 * self.out_channels + 5
        self.conv_offset = nn.Sequential(
            nn.Conv2d(input_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, feature, condition, trajectory_offset):
        output = self.conv_offset(condition)
        offset_x, offset_y, mask = torch.chunk(output, 3, dim=1)
        dcn_offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((offset_x, offset_y), dim=1))
        dcn_offset = dcn_offset + trajectory_offset.flip(1).repeat(
            1, dcn_offset.size(1) // 2, 1, 1)
        return torchvision.ops.deform_conv2d(
            feature,
            dcn_offset,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            torch.sigmoid(mask),
        )
