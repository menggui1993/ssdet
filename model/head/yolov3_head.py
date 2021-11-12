import torch.nn as nn
from model.base_module.conv_block import ConvBlock

class SingleHead(nn.Module):
    """
    Single detection head.
    conv 3x3x2n
    conv 1x1x(#classes+5)*3
    """
    def __init__(self,
                 num_classes,
                 in_channel,
                 norm_cfg,
                 act_cfg):
        super(SingleHead, self).__init__()
        self.out_channel = (num_classes+5)*3
        self.conv1 = ConvBlock(in_channel, in_channel*2, 3, padding=1, 
                        norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.out_conv = nn.Conv2d(in_channel*2, self.out_channel, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.out_conv(x)
        return x
        

class YoloHead(nn.Module):
    """
    Yolo head.
    Input feats from different scales
    Conv 3x3x2n
    Conv 1x1x((#class+5)*3)
    """
    def __init__(self,
                 num_classes,
                 in_channels=(512,256,128),
                 norm_cfg=dict(type='BN', args=dict()),
                 act_cfg=dict(type='LeakyRelu', args=dict(negative_slope=0.1))):
        super(YoloHead, self).__init__()
        self.num_scales = len(in_channels)
        
        self.heads = []
        for i,in_channel in enumerate(in_channels):
            module_name = "out_head_" + str(i)
            self.add_module(module_name,
                            SingleHead(num_classes, in_channel, norm_cfg, act_cfg))
            self.heads.append(module_name)
        
    def forward(self, feats):
        assert(len(feats) == self.num_scales)

        outs = []
        for i in range(self.num_scales):
            head = getattr(self, self.heads[i])
            out = head(feats[i])
            outs.append(out)

        return tuple(outs)

