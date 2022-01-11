import torch
import torch.nn as nn
import math
from model.base_module.conv_block import ConvBlock

class ScaleExp(nn.Module):
    def __init__(self,init_value=1.0):
        super(ScaleExp,self).__init__()
        self.scale=nn.Parameter(torch.tensor([init_value],dtype=torch.float32))
    def forward(self,x):
        return torch.exp(x*self.scale)

class FcosHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channel=256,
                 stacked_convs = 4,
                 num_strides = 5,
                 norm_cfg=dict(type='GN', args=dict(num_groups=32)),
                 act_cfg=dict(type='Relu', args=dict())):
        super(FcosHead, self).__init__()
        self.num_strides = num_strides
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
        self.scale_exp = nn.ModuleList([ScaleExp(1.0) for _ in range(num_strides)])

        self.init_weight()
        nn.init.constant_(self.cls_head.bias,-math.log((1 - 0.01) / 0.01))

    def forward(self, inputs):
        assert len(inputs) == self.num_strides
        cls_outs = []
        center_outs = []
        reg_outs = []
        for i, input in enumerate(inputs):
            cls_feat = self.cls_convs(input)
            reg_feat = self.reg_convs(input)
            cls_out = self.cls_head(cls_feat)
            center_out = self.center_head(cls_feat)
            reg_out = self.reg_head(reg_feat)
            reg_out = self.scale_exp[i](reg_out)
            cls_outs.append(cls_out)
            center_outs.append(center_out)
            reg_outs.append(reg_out)
        return cls_outs, center_outs, reg_outs

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
