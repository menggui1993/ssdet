import torch.nn as nn
from model.base_module.mish import Mish
from model.base_module.DCNv2.dcn_v2 import DCN

class ConvBlock(nn.Module):
    """
    Base convolution block.
    Convolution
    Normalization
    Activation
    """
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 norm_cfg=dict(type='BN', args=dict()),
                 act_cfg=dict(type='Relu', args=dict()),
                 dcn=False):
        super(ConvBlock, self).__init__()
        with_bias = norm_cfg is None
        if dcn:
            self.conv = DCN(in_channel, out_channel, (kernel_size,kernel_size),
                            stride, padding, dilation, groups)
        else:
            self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                            stride, padding, dilation, groups, with_bias)
        if norm_cfg is None:
            self.norm = None
        else:
            norm_type = norm_cfg['type']
            if norm_type == "BN":
                self.norm = nn.BatchNorm2d(out_channel, **norm_cfg['args'])
            elif norm_type == "GN":
                self.norm = nn.GroupNorm(num_groups=norm_cfg['args']['num_groups'], num_channels=out_channel)
            else:
                raise KeyError(f'norm type {norm_type} not supported')

        if act_cfg is None:
            self.act = None
        else:
            act_type = act_cfg['type']
            if act_type == "Relu":
                self.act = nn.ReLU(**act_cfg['args'])
            elif act_type == "LeakyRelu":
                self.act = nn.LeakyReLU(**act_cfg['args'])
            elif act_type == "PRelu":
                self.act = nn.PReLU(**act_cfg['args'])
            elif act_type == "Tanh":
                self.act = nn.Tanh(**act_cfg['args'])
            elif act_type == "Mish":
                self.act = Mish(**act_cfg['args'])
            else:
                raise KeyError(f'activation type {act_type} not supported')

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x    