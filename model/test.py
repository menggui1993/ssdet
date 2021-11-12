import torch
from model.backbone.darknet import Darknet
from model.neck.yolov3_neck import Yolov3Neck
from model.head.yolov3_head import YoloHead
from model.backbone.resnet import ResNet
from model.neck.ppyolo_pan import PPYoloPAN
from model.neck.fpn import FPN
from model.head.fcos_head import FcosHead

if __name__ == '__main__':
    # darknet53 = Darknet()
    # neck = Yolov3Neck()
    # head = YoloHead(80)

    # rand_input = torch.rand(1,3,416,416)
    # outs = darknet53(rand_input)
    # print(outs[0].shape)
    # print(outs[1].shape)
    # print(outs[2].shape)

    # outs = neck(outs)
    # print(outs[0].shape)
    # print(outs[1].shape)
    # print(outs[2].shape)

    # outs = head(outs)
    # print(outs[0].shape)
    # print(outs[1].shape)
    # print(outs[2].shape)

    # resnet50 = ResNet(vd=True, dcn_stages=(4,)).cuda()
    # rand_input = torch.rand(1,3,224,224).cuda()
    # outs = resnet50(rand_input)
    # print(outs[0].shape)
    # print(outs[1].shape)
    # print(outs[2].shape)

    # ppyolo_neck = PPYoloPAN().cuda()
    # outs = ppyolo_neck(outs)
    # print(outs[0].shape)
    # print(outs[1].shape)
    # print(outs[2].shape)

    # ppyolo_head = YoloHead(80, in_channels=(256,512,1024)).cuda()
    # outs = ppyolo_head(outs)
    # print(outs[0].shape)
    # print(outs[1].shape)
    # print(outs[2].shape)

    resnet50 = ResNet().cuda()
    rand_input = torch.rand(1, 3, 800, 800).cuda()
    outs = resnet50(rand_input)
    print("backbone")
    for out in outs:
        print(out.shape)

    fpn = FPN(num_out=5).cuda()
    outs = fpn(outs)
    print("fpn")
    for out in outs:
        print(out.shape)

    fcos_head = FcosHead(80).cuda()
    cls_outs, center_outs, reg_outs = fcos_head(outs)
    print("head")
    for cls_out in cls_outs:
        print(cls_out.shape)
    for center_out in center_outs:
        print(center_out.shape)
    for reg_out in reg_outs:
        print(reg_out.shape)