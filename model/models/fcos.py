import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.backbone.resnet import ResNet
from model.neck.fpn import FPN
from model.head.fcos_head import FcosHead
from model.assigner.gt_assign import generate_grid_points, fcos_assigner
from model.loss.focal_loss import FocalLoss
from model.loss.iou_loss import IouLoss
from model.util import distance2bbox

class Fcos(nn.Module):
    def __init__(self, num_classes):
        super(Fcos, self).__init__()
        self.backbone = ResNet()
        self.neck = FPN(num_out=5)
        self.head = FcosHead(num_classes)
        self.num_classes = num_classes
        self.strides = (8, 16, 32, 64, 128)
        self.conf_thres = 0.3

        self.cls_loss = FocalLoss()
        self.bbox_loss = IouLoss()
        self.center_loss = nn.BCEWithLogitsLoss()
        self.cls_weight = 1.0
        self.bbox_weight = 1.0
        self.center_weight = 1.0

    def forward(self, x):
        """
        outs: (cls_outs, center_outs, reg_outs)
        cls_outs' shape [(b,class,h1,w1), (b,class,h2,w2), ...]
        center_outs' shape [(b,1,h1,w1), (b,1,h2,w2), ...]
        reg_outs' shape [(b,4,h1,w1), (b,4,h2,w2), ...]
        """
        outs = self.backbone(x)
        outs = self.neck(outs)
        outs = self.head(outs)
        return outs

    def generate_points(self, input_w, input_h):
        """
        Generate anchor points

        Args:
            input_w: model input width
            input_h: model input height

        Returns:
            points: shape [(w1*h1, 2), (w2*h2, 2), ...]
        """
        points = []
        for stride in self.strides:
            w = math.ceil(input_w / stride)
            h = math.ceil(input_h / stride)
            points.append(generate_grid_points(h, w, stride))
        return points

    def calc_loss(self, cls_preds, center_preds, reg_preds, 
                    gt_bboxes, gt_labels, points):
        """
        Calculate loss

        Args:
            cls_preds: shape [(b,class,h1,w1), (b,class,h2,w2), ...]
            center_preds: shape [(b,1,h1,w1), (b,1,h2,w2), ...]
            reg_preds: shape [(b,4,h1,w1), (b,4,h2,w2), ...]
            gt_bboxes: shape (b, N, 4)
            gt_labels: shape (b, N)
            points: shape [(w1*h1, 2), (w2*h2, 2), ...]
        
        Returns:
            (cls_loss, center_loss, bbox_loss)
        """
        batch_size = gt_bboxes.shape[0]
        bg_class_id = self.num_classes
        cls_targets, center_targets, reg_targets = fcos_assigner(gt_bboxes, gt_labels, points, bg_class_id)

        mask_pos = (cls_targets.reshape(-1) != bg_class_id).nonzero().reshape(-1)
        # num_pos = mask_pos.shape[0]
        # print(num_pos)

        cls_targets = F.one_hot(cls_targets, num_classes=self.num_classes+1).float()
        cls_targets = cls_targets[:, :, :self.num_classes]
        cls_targets = cls_targets.reshape(-1, self.num_classes).to(cls_preds[0].device)
        center_targets = center_targets.reshape(-1).to(cls_preds[0].device)
        reg_targets = reg_targets.reshape(-1, 4).to(cls_preds[0].device)

        center_targets = center_targets[mask_pos]
        reg_targets = reg_targets[mask_pos]

        cls_preds = [cls_pred.reshape(batch_size, self.num_classes, -1) for cls_pred in cls_preds]
        cls_preds = torch.cat(cls_preds, dim=-1).permute(0, 2, 1).reshape(-1, self.num_classes)
        center_preds = [center_pred.reshape(batch_size, -1) for center_pred in center_preds]
        center_preds = torch.cat(center_preds, dim=-1).reshape(-1)
        reg_preds = [reg_pred.reshape(batch_size, 4, -1) for reg_pred in reg_preds]
        reg_preds = torch.cat(reg_preds, dim=-1).permute(0, 2, 1).reshape(-1, 4)

        center_preds = center_preds[mask_pos]
        reg_preds = reg_preds[mask_pos]
        pos_points = torch.cat(points).repeat(batch_size, 1)[mask_pos]

        bbox_preds = distance2bbox(pos_points, reg_preds)
        bbox_targets = distance2bbox(pos_points, reg_targets)

        cls_loss = self.cls_loss(cls_preds, cls_targets)
        center_loss =self.center_loss(center_preds, center_targets)
        bbox_loss = self.bbox_loss(bbox_preds, bbox_targets)

        return cls_loss, center_loss, bbox_loss

    def weighted_sum_loss(self, cls_loss, center_loss, bbox_loss):
        loss = self.cls_weight * cls_loss + self.center_weight * center_loss + self.bbox_weight * bbox_loss
        return loss

    def postprocess(self, cls_preds, center_preds, reg_preds, points):
        detections = []
        batch_size = cls_preds[0].shape[0]
        anchor_pts = torch.cat(points)
        for b in range(batch_size):
            cls_outs = [cls_pred[b].reshape(-1, self.num_classes) for cls_pred in cls_preds]
            center_outs = [center_pred[b].reshape(-1) for center_pred in center_preds]
            reg_outs = [reg_pred[b].reshape(-1, 4) for reg_pred in reg_preds]
            cls_outs = torch.cat(cls_outs, dim=0)
            center_outs = torch.cat(center_outs, dim=0)
            reg_outs = torch.cat(reg_outs, dim=0)

            max_scores, max_idx = torch.max(cls_outs, dim=1)
            scores = max_scores * center_outs
            mask_pos = ((max_idx != 0) & (scores > self.conf_thres))
            offsets = reg_outs[mask_pos]
            classes = max_idx[mask_pos]
            scores = scores[mask_pos]
            pts = anchor_pts[mask_pos]
            # print(bboxes.shape)
            # print(classes.shape)
            # print(scores.shape)
        
            left = pts[:, 0] - offsets[:, 0]
            top = pts[:, 1] - offsets[:, 1]
            right = pts[:, 0] + offsets[:, 2]
            bottom = pts[:, 1] + offsets[:, 3]
            objs = torch.stack((left, top, right, bottom, classes, scores), dim=-1)
            # print(objs.shape)
            detections.append(objs)

        return detections


