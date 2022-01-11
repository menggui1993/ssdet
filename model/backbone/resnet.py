import torch
import torch.nn as nn
from model.base_module.conv_block import ConvBlock

class Bottleneck(nn.Module):
    """
    Basic residual block in Resnet50. The block structure is
    Conv    out_ch/4    1x1
    Conv    out_ch/4    3x3     stride
    Conv    out_ch      1x1
    Residual
    """
    def __init__(self,
                 in_channel,
                 out_channel,
                 stride = 1,
                 vd = False,
                 dcn = False):
        super(Bottleneck, self).__init__()
        norm_cfg = dict(type='BN', args=dict())
        act_cfg = dict(type='Relu', args=dict())
        self.conv1 = ConvBlock(in_channel, out_channel//4, 1, 
                            norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvBlock(out_channel//4, out_channel//4, 3, stride, 1,
                            norm_cfg=norm_cfg, act_cfg=act_cfg, dcn=dcn)
        self.conv3 = ConvBlock(out_channel//4, out_channel, 1,
                            norm_cfg=norm_cfg, act_cfg=None)
        self.downsample = None
        if stride != 1 or in_channel != out_channel:
            if vd == True and stride == 2:
                self.downsample = nn.Sequential(nn.AvgPool2d(2, 2, 0, ceil_mode=True),
                                                ConvBlock(in_channel, out_channel, 1, 1))
            else:
                self.downsample = ConvBlock(in_channel, out_channel, 1, stride,
                                    norm_cfg=norm_cfg, act_cfg=None)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        
        out = self.act(out)
        return out


class ResStage(nn.Module):
    """
    ResNet is constructed by several stages.
    Each stage is constructed by a BottleNeck that changes number of channels
    following by multiple BottleNecks.
    """
    def __init__(self, 
                 block_count,
                 in_channel,
                 out_channel,
                 stride,
                 vd = False,
                 dcn = False):
        super(ResStage, self).__init__()
        self.block_count = block_count
        self.res_blocks = []
        self.add_module("res_block_0", 
                    Bottleneck(in_channel, out_channel, stride, vd=vd, dcn=dcn))
        self.res_blocks.append("res_block_0")

        for i in range(1, block_count):
            block_name = "res_block_" + str(i)
            self.add_module(block_name,
                    Bottleneck(out_channel, out_channel, 1, vd=vd, dcn=dcn))
            self.res_blocks.append(block_name)
    
    def forward(self, x):
        for block_name in self.res_blocks:
            block = getattr(self, block_name)
            x = block(x)
        return x


class ResNet(nn.Module):
    arch_settings = {
        50: ((3, 4, 6, 3), (256, 512, 1024, 2048), (1, 2, 2, 2))
    }
    def __init__(self,
                 out_stages = (2, 3, 4),
                 vd = False,
                 dcn_stages = (),
                 pretrained=True):
        super(ResNet, self).__init__()
        self.out_stages = out_stages
        self.vd = vd
        self.block_counts, self.block_channels, self.block_strides = self.arch_settings[50]
        norm_cfg = dict(type='BN', args=dict())
        act_cfg = dict(type='Relu', args=dict())
        
        if not self.vd:
            self.conv1 = ConvBlock(3, 64, 7, 2, 3, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.conv1_1 = ConvBlock(3, 32, 3, 2, 1)
            self.conv1_2 = ConvBlock(32, 32, 3, 1, 1)
            self.conv1_3 = ConvBlock(32, 64, 3, 1, 1)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        # define stage blocks
        cur_channel = 64
        self.stages = []
        for i, b_count in enumerate(self.block_counts):
            module_name = "stage_" + str(i+1)
            dcn = False
            if i+1 in dcn_stages:
                dcn = True
            self.add_module(module_name, 
                ResStage(b_count, cur_channel, self.block_channels[i], self.block_strides[i], vd=vd, dcn=dcn))
            self.stages.append(module_name)
            cur_channel = self.block_channels[i]
        if pretrained:
            self.load_state_dict(torch.load('checkpoint/resnet50.pth'))
        else:
            self.init_weight()
        
    def forward(self, x):
        outs = []
        if self.vd:
            x = self.conv1_1(x)
            x = self.conv1_2(x)
            x = self.conv1_3(x)
        else:
            x = self.conv1(x)
        x = self.maxpool(x)
        
        for i, module_name in enumerate(self.stages):
            block = getattr(self, module_name)
            x = block(x)
            if i+1 in self.out_stages:
                outs.append(x)
        return tuple(outs)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    model = ResNet()
    # model.load_state_dict(torch.load('resnet50.pth'))