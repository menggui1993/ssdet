import torch
import torch.nn as nn
import math

# def bbox_iou(bboxes1, bboxes2, iou_type='iou'):
#     '''
#     bboxes1: [N, 4]
#     bboxes2: [N, 4]
#     '''
#     area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
#     area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

#     lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])  # [B,N,M,2]
#     rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])  # [B,N,M,2]
#     wh = (rb - lt).clamp(min=0)

#     inter = wh[..., 0] * wh[..., 1]           # [B,N,M]
#     union = area1[..., None] + area2[..., None, :] - inter      # [B,N,M]
#     iou = inter / union

#     if iou_type == 'giou' or iou_type == 'diou' or iou_type == 'ciou':
#         cw = torch.max(bboxes1[..., :, None, 2], bboxes2[..., None, :, 2]) - torch.min(bboxes1[..., :, None, 0], bboxes2[..., None, :, 0])
#         ch = torch.max(bboxes1[..., :, None, 3], bboxes2[..., None, :, 3]) - torch.min(bboxes1[..., :, None, 1], bboxes2[..., None, :, 1])
#         if iou_type == 'giou':
#             c_area = cw * ch
#             giou = iou - (c_area - union) / c_area
#             return giou
#         if iou_type == 'diou' or iou_type == 'ciou':
#             c2 = cw ** 2 + ch ** 2
#             d2 = ((bboxes2[..., None, :, 0] + bboxes2[..., None, :, 2]) - (bboxes1[..., :, None, 0] + bboxes1[..., :, None, 2])) ** 2 / 4 + ((bboxes2[..., None, :, 1] + bboxes2[..., None, :, 3]) - (bboxes1[..., :, None, 1] + bboxes1[..., :, None, 3])) ** 2 / 4
#             if iou_type == 'diou':
#                 diou = iou - d2 / c2
#                 return diou
#             if iou_type == 'ciou':
#                 w1 = bboxes1[..., 2] - bboxes1[..., 0]
#                 h1 = bboxes1[..., 3] - bboxes1[..., 1]
#                 w2 = bboxes2[..., 2] - bboxes2[..., 0]
#                 h2 = bboxes2[..., 3] - bboxes2[..., 1]
#                 v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2[..., None, :] / h2[..., None, :]) - torch.atan(w1[..., :, None] / h1[..., :, None]), 2)
#                 alpha = v / (1 - iou +v)
#                 ciou = iou - (d2 / c2 + alpha * v)
#                 return ciou

#     return iou

def bbox_iou(bboxes1, bboxes2, iou_type='iou'):
    '''
    bboxes1: [N, 4]
    bboxes2: [N, 4]
    '''
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [N,2]
    rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [N,2]
    wh = (rb - lt).clamp(min=0)

    inter = wh[:, 0] * wh[:, 1]           
    union = area1 + area2 - inter
    iou = inter / union

    if iou_type == 'giou' or iou_type == 'diou' or iou_type == 'ciou':
        cw = torch.max(bboxes1[:, 2], bboxes2[:, 2]) - torch.min(bboxes1[:, 0], bboxes2[:, 0])
        ch = torch.max(bboxes1[:, 3], bboxes2[:, 3]) - torch.min(bboxes1[:, 1], bboxes2[:, 1])
        if iou_type == 'giou':
            c_area = cw * ch
            giou = iou - (c_area - union) / c_area
            return giou
        if iou_type == 'diou' or iou_type == 'ciou':
            c2 = cw ** 2 + ch ** 2
            d2 = ((bboxes2[:, 0] + bboxes2[:, 2]) - (bboxes1[:, 0] + bboxes1[:, 2])) ** 2 / 4 + ((bboxes2[:, 1] + bboxes2[:, 3]) - (bboxes1[:, 1] + bboxes1[:, 3])) ** 2 / 4
            if iou_type == 'diou':
                diou = iou - d2 / c2
                return diou
            if iou_type == 'ciou':
                w1 = bboxes1[:, 2] - bboxes1[:, 0]
                h1 = bboxes1[:, 3] - bboxes1[:, 1]
                w2 = bboxes2[:, 2] - bboxes2[:, 0]
                h2 = bboxes2[:, 3] - bboxes2[:, 1]
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                alpha = v / (1 - iou +v)
                ciou = iou - (d2 / c2 + alpha * v)
                return ciou

    return iou


class IouLoss(nn.Module):
    def __init__(self, 
                 iou_type='iou',    #[iou, ciou, diou, giou]
                 mode='log',        #['linear', 'log'] 
                 reduction='mean'
                 ):
        super(IouLoss, self).__init__()
        self.iou_type = iou_type
        self.mode = mode
        self.reduction = reduction

    def forward(self, inputs, targets):
        iou = bbox_iou(inputs, targets, iou_type=self.iou_type).clamp(min=1e-6)
        if self.mode == 'linear':
            loss = 1 - iou
        elif self.mode == 'log':
            loss = -torch.log(iou)
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss


if __name__ == "__main__":
    bboxes1 = torch.rand(8, 5, 4)
    bboxes2 = torch.rand(8, 10, 4)
    iou = bbox_iou(bboxes1, bboxes2, 'iou')
    print(iou.shape)
    