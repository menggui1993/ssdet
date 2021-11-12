import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_module.conv_block import ConvBlock
from model.base_module.misc import SPP, DropBlock2D


class PPYoloDetBlock(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 with_spp=False,
                 act_cfg = dict(type='Mish', args=dict())
                 ):
        super(PPYoloDetBlock, self).__init__()
        self.conv1 = ConvBlock(in_ch, out_ch//2, 1, act_cfg=act_cfg)
        self.conv2 = ConvBlock(out_ch//2, out_ch//2, 1, act_cfg=act_cfg)
        self.conv3 = ConvBlock(out_ch//2, out_ch//2, 3, 1, 1, act_cfg=act_cfg)
        self.conv4 = ConvBlock(out_ch//2, out_ch//2, 1, act_cfg=act_cfg)
        self.spp = None
        if with_spp:
            self.spp = SPP(out_ch//2, act_cfg=act_cfg)
        else:
            self.conv5 = ConvBlock(out_ch//2, out_ch//2, 3, 1, 1, act_cfg=act_cfg)
        self.dropblock = DropBlock2D(drop_prob=0.9, block_size=3)
        self.conv6 = ConvBlock(out_ch//2, out_ch//2, 1, act_cfg=act_cfg)
        self.conv7 = ConvBlock(out_ch//2, out_ch//2, 3, 1, 1, act_cfg=act_cfg)

        self.res_conv = ConvBlock(in_ch, out_ch//2, 1, act_cfg=act_cfg)
        self.last_conv = ConvBlock(out_ch, out_ch, 1, act_cfg=act_cfg)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        if self.spp is not None:
            out = self.spp(out)
        else:
            out = self.conv5(out)
        out = self.dropblock(out)
        out = self.conv6(out)
        out = self.conv7(out)

        residual = self.res_conv(x)
        out = torch.cat((out, residual), dim=1)
        out = self.last_conv(out)
        return out

class PPYoloUpsample(nn.Module):
    def __init__(self,
                 in_ch,
                 act_cfg = dict(type='Mish', args=dict())):
        super(PPYoloUpsample, self).__init__()
        self.conv = ConvBlock(in_ch, in_ch, 1, act_cfg=act_cfg)
        self.upsample = nn.Upsample(scale_factor=2)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.upsample(out)
        return out

class PPYoloPAN(nn.Module):
    def __init__(self,
                 in_channels = (512, 1024, 2048),
                 out_channels = (256, 512, 1024),
                 act_cfg = dict(type='Mish', args=dict())):
        super(PPYoloPAN, self).__init__()
        self.num_scales = len(in_channels)

        self.fpn_block_c5 = PPYoloDetBlock(2048, 1024, with_spp=True, act_cfg=act_cfg)
        self.upsample_c5 = PPYoloUpsample(1024, act_cfg=act_cfg)
        self.fpn_block_c4 = PPYoloDetBlock(2048, 512, act_cfg=act_cfg)
        self.upsample_c4 = PPYoloUpsample(512, act_cfg=act_cfg)
        self.fpn_block_c3 = PPYoloDetBlock(1024, 256, act_cfg=act_cfg)

        self.downsample_c3 = ConvBlock(256, 256, 3, 2, 1, act_cfg=act_cfg)
        self.pan_block_c4 = PPYoloDetBlock(768, 512, act_cfg=act_cfg)
        self.downsample_c4 = ConvBlock(512, 512, 3, 2, 1, act_cfg=act_cfg)
        self.pan_block_c5 = PPYoloDetBlock(1536, 1024, act_cfg=act_cfg)

    def forward(self, feats):
        assert len(feats) == self.num_scales
        outs = []
        c3_feat, c4_feat, c5_feat = feats
        fpn_out_c5 = self.fpn_block_c5(c5_feat)
        cat_c4 = torch.cat((c4_feat, self.upsample_c5(fpn_out_c5)), dim=1)
        fpn_out_c4 = self.fpn_block_c4(cat_c4)
        cat_c3 = torch.cat((c3_feat, self.upsample_c4(fpn_out_c4)), dim=1)
        fpn_out_c3 = self.fpn_block_c3(cat_c3)

        outs.append(fpn_out_c3)
        cat_c4 = torch.cat((fpn_out_c4, self.downsample_c3(fpn_out_c3)), dim=1)
        pan_out_c4 = self.pan_block_c4(cat_c4)
        outs.append(pan_out_c4)
        cat_c5 = torch.cat((fpn_out_c5, self.downsample_c4(pan_out_c4)), dim=1)
        pan_out_c5 = self.pan_block_c5(cat_c5)
        outs.append(pan_out_c5)

        return tuple(outs)