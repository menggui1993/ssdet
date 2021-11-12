import torch
import torch.nn as nn
from model.base_module.conv_block import ConvBlock

class FcosHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channel=256,
                 stacked_convs = 4,
                 norm_cfg=dict(type='GN', args=dict(num_groups=32)),
                 act_cfg=dict(type='Relu', args=dict())):
        super(FcosHead, self).__init__()
        cls_convs = []
        reg_convs = []
        for i in range(stacked_convs):
            cls_convs.append(ConvBlock(in_channel, in_channel, 3, padding=1,
                                        norm_cfg=norm_cfg, act_cfg=act_cfg))
            reg_convs.append(ConvBlock(in_channel, in_channel, 3, padding=1, 
                                        norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.cls_convs = nn.Sequential(*cls_convs)
        self.reg_convs = nn.Sequential(*reg_convs)
        self.cls_head = nn.Conv2d(in_channel, num_classes, 3, padding=1)
        self.center_head = nn.Conv2d(in_channel, 1, 3, padding=1)
        self.reg_head = nn.Conv2d(in_channel, 4, 3, padding=1)

    def forward(self, inputs):
        cls_outs = []
        center_outs = []
        reg_outs = []
        for input in inputs:
            cls_feat = self.cls_convs(input)
            reg_feat = self.reg_convs(input)
            cls_out = self.cls_head(cls_feat)
            center_out = self.center_head(cls_feat)
            reg_out = self.reg_head(reg_feat)
            cls_outs.append(cls_out)
            center_outs.append(center_out)
            reg_outs.append(reg_out)
        return cls_outs, center_outs, reg_outs
        
