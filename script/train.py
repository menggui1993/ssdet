import torch
import torch.nn as nn
import argparse
from model.models.fcos import Fcos
from data.dataset import YoloDataset
from data.transforms import Compose, Resize, RandomHorizontalFlip, Normalize, ToTorchTensor

batch_size = 8
num_classes = 5
input_w, input_h = (400, 400)
gt_bboxes = torch.tensor([[20, 40, 64, 80],
                            [200, 100, 300, 150],
                            [200, 20, 380, 80]])
gt_labels = torch.tensor([[3], [2], [1]])

gt_bboxes = gt_bboxes.repeat(batch_size, 1, 1)
gt_labels = gt_labels.repeat(batch_size, 1, 1).squeeze(-1)

fcos_model = Fcos(num_classes)
points = fcos_model.generate_points(input_w, input_h)
torch.manual_seed(666)
input = torch.rand(batch_size, 3, input_h, input_w)
print(input.shape)
print(gt_bboxes.shape)
print(gt_labels.shape)
cls_preds, center_preds, reg_preds = fcos_model(input)
cls_loss, center_loss, bbox_loss = fcos_model.calc_loss(cls_preds, center_preds, reg_preds, 
                gt_bboxes, gt_labels, points)

loss = fcos_model.weighted_sum_loss(cls_loss, center_loss, bbox_loss)
print(cls_loss)
print(center_loss)
print(bbox_loss)
print(loss)

# detections = fcos_model.postprocess(cls_preds, center_preds, reg_preds, points)
# print(detections[0].shape)
# print(detections[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=8, type=int)

    