import torch
import numpy as np

def calcIOU(bbox_pred, bbox_gt):
    left = torch.max(bbox_pred[:, None, 0], bbox_gt[None, :, 0])
    top = torch.max(bbox_pred[:, None, 1], bbox_gt[None, :, 1])
    right = torch.min(bbox_pred[:, None, 2], bbox_gt[None, :, 2])
    bottom = torch.min(bbox_pred[:, None, 3], bbox_gt[None, :, 3])

    inter = (right - left) * (bottom - top)
    union = (bbox_pred[:, 2] - bbox_pred[:, 0]) * (bbox_pred[:, 3] - bbox_pred[:, 1])
    union = union[:, None] - inter
    iou = inter / union
    return iou

def eval(detections, cls_gts, bbox_gts, classes, iou_thres=0.5):
    batch_size = len(detections)
    # cls_gts = cls_gts.cpu()
    # bbox_gts = bbox_gts.cpu()
    
    cls_preds = []
    score_preds = []
    tps = []
    fps = []

    # check every image
    for b in range(batch_size):
        detection = detections[b].cpu()
        cls_gt = cls_gts[b].cpu()
        bbox_gt = bbox_gts[b].cpu()

        num_pred = detection.shape[0]
        if num_pred == 0:
            continue

        bbox_pred = detection[:, :4]
        cls_pred = detection[:, 4]
        score_pred = detection[:, 5]

        print(detection)
        print(bbox_gt)

        num_gt = cls_gt.shape[0]
        if num_gt == 0:
            tp = torch.zeros(cls_pred.shape)
            fp = torch.ones(cls_pred.shape)
        else:
            # calculate iou between all predict bboxes and gt bboxes
            iou = calcIOU(bbox_pred, bbox_gt)
            iou[cls_pred[:, None] != cls_gt[None, :]] = 0   # ignore if classes not same

            max_iou, max_gt_idx = torch.max(iou, dim=1)     # find best gt for each predict bbox
            tp = torch.zeros(cls_pred.shape)
            fp = torch.zeros(cls_pred.shape)
            # if iou > iou_threshold, then true positive, else false positive
            tp[max_iou > iou_thres] = 1
            fp[max_iou < iou_thres] = 1

        cls_preds.append(cls_pred)
        score_preds.append(score_pred)
        tps.append(tp)
        fps.append(fp)
    
    # combine all predictions
    if len(cls_preds) == 0:
        return 0.0

    cls_preds = torch.cat(cls_preds)
    score_preds = torch.cat(score_preds)
    tps = torch.cat(tps)
    fps = torch.cat(fps)

    # calculate ap for each class
    aps = []
    for c in range(classes):
        npos = sum([torch.count_nonzero(cls_gt.cpu() == c) for cls_gt in cls_gts])
        # npos = torch.count_nonzero(cls_gts == c)

        c_score_preds = score_preds[cls_preds == c]
        c_tps = tps[cls_preds == c]
        c_fps = fps[cls_preds == c]
        
        # sorted based on confidence
        sorted_idx = torch.argsort(c_score_preds, descending=True)
        c_tps = c_tps[sorted_idx]
        c_fps = c_fps[sorted_idx]

        tp = torch.cumsum(c_tps, dim=0)
        fp = torch.cumsum(c_fps, dim=0)
        
        if tp.shape[0] == 0:
            recall = torch.zeros(1)
            precision = torch.zeros(1)
        else:
            recall = tp / (npos + 1e-5)
            precision = tp / (tp + fp + 1e-5)

        recall = torch.cat((torch.zeros(1), recall, torch.ones(1)), dim=0)
        precision = torch.cat((torch.zeros(1), precision, torch.zeros(1)), dim=0)
        
        for i in range(precision.shape[0] - 1, 0, -1):
            precision[i - 1] = torch.maximum(precision[i - 1], precision[i])
        
        index = torch.nonzero(recall[1:] == recall[:-1])
        ap = torch.sum((recall[index + 1] - recall[index]) * precision[index + 1])
        aps.append(ap)
    # print(max(aps))
    return aps


        

        