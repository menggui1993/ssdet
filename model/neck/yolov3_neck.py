import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_module.conv_block import ConvBlock

class YoloNeckConv(nn.Module):
    """
    Each feature map will go through 5 conv with shape
    1x1xn, 3x3x2n, 1x1xn, 3x3x2n, 1x1xn
    """
    def __init__(self,
                 in_channel,
                 out_channel,
                 norm_cfg,
                 act_cfg):
        super(YoloNeckConv, self).__init__()
        self.conv1 = ConvBlock(in_channel, out_channel, 1, 
                        norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvBlock(out_channel, out_channel*2, 3, padding=1,
                        norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv3 = ConvBlock(out_channel*2, out_channel, 1,
                        norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv4 = ConvBlock(out_channel, out_channel*2, 3, padding=1,
                        norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv5 = ConvBlock(out_channel*2, out_channel, 1,
                        norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class YoloNeckUpsample(nn.Module):
    """
    Go through one 1x1xn/2 conv and upsample, 
    then concat with feature map from backbone
    """
    def __init__(self,
                 in_channel,
                 norm_cfg,
                 act_cfg):
        super(YoloNeckUpsample, self).__init__()
        self.conv = ConvBlock(in_channel, in_channel//2, 1,
                        norm_cfg=norm_cfg, act_cfg=act_cfg)
    
    def forward(self, x, ori_feat):
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat((x, ori_feat), 1)
        return x
        

class Yolov3Neck(nn.Module):
    """
    Yolov3 neck.
    Input top-down feature maps.
    Go bottom-up by conv and upsampling and 
    concat with corresponding input feature maps
    """
    def __init__(self, 
                 in_channels=(256,512,1024),
                 out_channels=(512,256,128),
                 norm_cfg = dict(type='BN', args=dict()),
                 act_cfg = dict(type='LeakyRelu', args=dict(negative_slope=0.1))):
        super(Yolov3Neck, self).__init__()
        self.num_scales = len(in_channels)
        
        self.conv_blocks = []
        last_out_ch = 0
        for i in range(self.num_scales):
            module_name = "conv_blocks_" + str(i)
            self.add_module(module_name,
                YoloNeckConv(in_channels[self.num_scales-i-1]+last_out_ch//2, 
                        out_channels[i], norm_cfg, act_cfg))
            self.conv_blocks.append(module_name)
            last_out_ch = out_channels[i]
       
        self.upsample_blocks = []
        for i in range(self.num_scales-1):
            module_name = "upsample_" + str(i)
            self.add_module(module_name,
                YoloNeckUpsample(out_channels[i], norm_cfg, act_cfg))
            self.upsample_blocks.append(module_name)
            
    def forward(self, feats):
        assert len(feats) == self.num_scales
        outs = []

        # first layer, no concat
        cblock = getattr(self, self.conv_blocks[0])
        x = cblock(feats[-1])
        outs.append(x)

        for i in range(self.num_scales-1):
            upsample = getattr(self, self.upsample_blocks[i])
            x = upsample(x, feats[self.num_scales-i-2])
            cblock = getattr(self, self.conv_blocks[i+1])
            x = cblock(x)
            outs.append(x)
        
        return tuple(outs)
