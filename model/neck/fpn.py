import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_module.conv_block import ConvBlock

class FPN(nn.Module):
    """
    FPN neck
    Args:
        extra_downsample: downsample method for extra outputs, oneof ['conv', 'pool']
    """
    def __init__(self, 
                 in_channels=(512,1024,2048),
                 out_channel=256,
                 num_out=3,
                 extra_downsample='conv',
                 norm_cfg=None,
                 act_cfg=None):
        super(FPN, self).__init__()
        self.num_in = len(in_channels)
        self.num_out = num_out
        self.project_convs = nn.ModuleList()
        self.out_convs = nn.ModuleList()
        for in_channel in in_channels:
            project_conv = ConvBlock(in_channel, out_channel, 1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
            self.project_convs.append(project_conv)
            out_conv = ConvBlock(out_channel, out_channel, 3, padding=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
            self.out_convs.append(out_conv)
        # more outputs, downsample from the last out stage
        if num_out > self.num_in:
            self.extra_downs = nn.ModuleList()
            for i in range(num_out - self.num_in):
                if extra_downsample == 'conv':
                    self.extra_downs.append(ConvBlock(out_channel, out_channel, 3, stride=2, padding=1,
                                    norm_cfg=norm_cfg, act_cfg=act_cfg))
                elif extra_downsample == 'pool':
                    self.extra_downs.append(nn.MaxPool2d(1, 2))
                else:
                    raise KeyError(f'down_sample type {extra_downsample} not supported')
        self.init_weight()

    def forward(self, inputs):
        assert len(inputs) == self.num_in
        outs = []
        proj_feats = []
        # convolution
        for i, project_conv in enumerate(self.project_convs):
            proj_feats.append(project_conv(inputs[i]))
        # upsample
        for i in range(self.num_in - 2, 0, -1):
            proj_feats[i] += F.interpolate(proj_feats[i+1], 
                    size=(proj_feats[i].shape[2], proj_feats[i].shape[3]), mode='nearest')
        for i, out_conv in enumerate(self.out_convs):
            out = out_conv(proj_feats[i])
            outs.append(out)
        # extra outputs
        if self.num_out > self.num_in:
            for extra_down in self.extra_downs:
                out = extra_down(outs[-1])
                outs.append(out)
        
        return tuple(outs)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)