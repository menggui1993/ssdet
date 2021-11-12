import torch
import torch.nn as nn
from model.backbone.darknet import Darknet
from model.neck.yolov3_neck import Yolov3Neck
from model.head.yolov3_head import YoloHead


class YoloV3(nn.Module):
    
    def __init__(self, 
                 num_classes,
                 strides = (8, 16, 32),
                 input_size = 416):
        super(YoloV3, self).__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.input_size = input_size
        
        self.backbone = Darknet()
        self.neck = Yolov3Neck()
        self.head = YoloHead(num_classes)

    def forward(self, x):
        outs = self.backbone(x)
        outs = self.neck(outs)
        outs = self.head(outs)
        return outs

    def calc_loss(self, preds, tgts):
        pass

    def gen_grid_anchor(self):
        anchors = [[10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],  [156,198],  [373,326]]
        self.grid_anchors = []
        for i in range(3):
            stride = self.strides[i]
            feat_size = self.input_size // stride
            anchor_map = torch.zeros(3, feat_size, feat_size, 4)
            anchor_map[:,:,:,0] = torch.arange(feat_size).repeat(3, feat_size, 1)
            anchor_map[:,:,:,1] = torch.arange(feat_size).repeat(3, feat_size, 1).permute(0,2,1)
            grid_wh = [torch.Tensor([w/stride, h/stride]) for w,h in anchors[i*3:i*3+3]]
            anchor_map[0,:,:,2:] = grid_wh[0]
            anchor_map[1,:,:,2:] = grid_wh[1]
            anchor_map[2,:,:,2:] = grid_wh[2]
            self.grid_anchors.append(anchor_map)
        
        
            

if __name__ == "__main__":
    yolov3 = YoloV3(80)
    yolov3.gen_grid_anchor()