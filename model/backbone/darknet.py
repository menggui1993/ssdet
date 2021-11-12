import torch
import torch.nn as nn
from model.base_module.conv_block import ConvBlock

class ResBlock(nn.Module):
    """
    Basic residual block in Darknet. The block structure is
    Conv    in_ch/2 1x1
    Conv    in_ch   3x3
    Residual
    """
    def __init__(self, 
                 in_channel,
                 norm_cfg,
                 act_cfg):
        super(ResBlock, self).__init__()
        self.conv1 = ConvBlock(in_channel, in_channel//2, 1,
                            norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvBlock(in_channel//2, in_channel, 3, padding=1,
                            norm_cfg=norm_cfg, act_cfg=act_cfg)
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        return x


class MultiResBlock(nn.Module):
    """
    Darknet is constructed by several MultiResBlock.
    Each MultiResBlock is constructed by a 3x3, downsample Conv 
    following by multiple ResBlock with same setting.
    """
    def __init__(self, 
                 block_count,
                 out_channel,
                 norm_cfg,
                 act_cfg):
        super(MultiResBlock, self).__init__()
        self.block_count = block_count
        self.downsample = ConvBlock(out_channel//2, out_channel, 3, 2, padding=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.res_blocks = []
        for i in range(block_count):
            block_name = "res_block_" + str(i)
            self.add_module(block_name,
                    ResBlock(out_channel, norm_cfg, act_cfg))
            self.res_blocks.append(block_name)
    
    def forward(self, x):
        x = self.downsample(x)
        for block_name in self.res_blocks:
            block = getattr(self, block_name)
            x = block(x)
        return x


class Darknet(nn.Module):
    """
    Darknet53
    ref: https://arxiv.org/abs/1804.02767
    
    """
    # arch_name: res_block_count, res_block_channel
    arch_settings = {
        53: ((1, 2, 8, 8, 4), (64, 128, 256, 512, 1024))
    }

    def __init__(self,
                 out_stages = (3, 4, 5),
                 norm_cfg = dict(type='BN', args=dict()),
                 act_cfg = dict(type='LeakyRelu', args=dict(negative_slope=0.1))):
        super(Darknet, self).__init__()
        self.out_stages = out_stages
        self.block_counts, self.block_channels = self.arch_settings[53]
        
        self.stages = []
        # define first convolution
        self.conv1 = ConvBlock(3, 32, 3, padding=1, 
                            norm_cfg=norm_cfg, act_cfg=act_cfg)
        # define stage blocks
        for i, b_count in enumerate(self.block_counts):
            module_name = "stage_" + str(i+1)
            self.add_module(module_name, 
                MultiResBlock(b_count, self.block_channels[i], norm_cfg, act_cfg))
            self.stages.append(module_name)
                
    def forward(self, x):
        outs = []
        # first convolution
        x = self.conv1(x)
        # 5 stage
        for i, module_name in enumerate(self.stages):
            block = getattr(self, module_name)
            x = block(x)
            if i+1 in self.out_stages:
                outs.append(x)
        
        return tuple(outs)

if __name__ == '__main__':
    darknet53 = Darknet()
    rand_input = torch.rand(1,3,416,416)
    outs = darknet53(rand_input)
    print(outs[0].shape)
    print(outs[1].shape)
    print(outs[2].shape)