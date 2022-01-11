import torch
from torch import optim
from torch.utils.data import DataLoader

import yaml
import argparse
import time
import os
from tqdm import tqdm

from model.models.fcos import Fcos
from data.dataset import YoloDataset
from data.transforms import Compose, Resize, RandomHorizontalFlip, Normalize, ToTorchTensor
from evaluation.eval import eval
from optimizer.warmup import WarmupLR


def get_total_and_free_memory_in_Mb(cuda_device):
    devices_info_str = os.popen(
        "nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader"
    )
    devices_info = devices_info_str.read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(",")
    return int(total), int(used)

def occupy_mem(cuda_device, mem_ratio=0.9):
    """
    pre-allocate gpu memory for training to avoid memory Fragmentation.
    """
    total, used = get_total_and_free_memory_in_Mb(cuda_device)
    max_mem = int(total * mem_ratio)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256, 1024, block_mem)
    del x
    time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str, default='config/fcos.yaml')
    args = parser.parse_args()

    f = open(args.cfg)
    opt = yaml.load(f)

    # setup data
    train_datasets = YoloDataset(opt['train_img_dir'], opt['train_label_dir'], opt['classes'])
    train_dataloader = DataLoader(train_datasets, opt['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    val_datasets = YoloDataset(opt['val_img_dir'], opt['val_label_dir'], opt['classes'])
    val_dataloader = DataLoader(val_datasets, opt['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
    num_classes = train_datasets.get_num_classes()

    # init model
    model = Fcos(num_classes)
    model = model.cuda()
    anchor_points = model.generate_points(640, 640)
    anchor_points = [points.cuda() for points in anchor_points]

    # setup optimizer
    optimizer = optim.SGD(model.parameters(), opt['lr'], momentum=0.9, weight_decay=0.0001)
    ms_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, opt['lr_steps'], gamma=0.1)
    lr_scheduler = WarmupLR(optimizer, ms_scheduler, opt['warmup_steps'])

    num_epochs = opt['epochs']
    num_batch = 0
    for epoch in range(num_epochs):
        print('-' * 40)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        # train
        epoch_detections = []
        epoch_cls_gts = []
        epoch_bbox_gts = []
        sum_loss = 0.
        sum_cls_loss = 0.
        sum_center_loss = 0.
        sum_bbox_loss = 0.
        count = 0
        for data in tqdm(train_dataloader):
            # print('lr: ', optimizer.param_groups[0]['lr'])
            imgs, cls_targets, bbox_targets = data
            imgs = imgs.cuda()
            cls_targets = cls_targets.cuda()
            bbox_targets = bbox_targets.cuda()
            optimizer.zero_grad()
            cls_preds, center_preds, reg_preds = model(imgs)
            cls_loss, center_loss, bbox_loss = model.calc_loss(cls_preds, center_preds, reg_preds, 
                        bbox_targets, cls_targets, anchor_points)
            loss = model.weighted_sum_loss(cls_loss, center_loss, bbox_loss)
            loss.backward()
            optimizer.step()

            sum_loss += loss
            sum_cls_loss += cls_loss
            sum_center_loss += center_loss
            sum_bbox_loss += bbox_loss
            count += 1

            detections = model.postprocess(cls_preds, center_preds, reg_preds, anchor_points, 640, 416)
            print(detections)
            for i in range(train_dataloader.batch_size):
                img_cls_targets = cls_targets[i]
                img_bbox_targets = bbox_targets[i]
                pos_id = torch.nonzero(img_cls_targets != num_classes)
                epoch_detections.append(detections[i])
                epoch_cls_gts.append(img_cls_targets[pos_id].reshape(-1))
                epoch_bbox_gts.append(img_bbox_targets[pos_id].reshape(-1, 4))
            
            if num_batch < opt['warmup_steps']:
                lr_scheduler.step()
            num_batch += 1
        
        sum_loss /= count
        sum_cls_loss /= count
        sum_center_loss /= count
        sum_bbox_loss /= count
        print("loss: %.2f, cls loss: %.2f, center loss: %.2f, iou loss: %.2f"%(sum_loss, sum_cls_loss, sum_center_loss, sum_bbox_loss))
        ap = eval(epoch_detections, epoch_cls_gts, epoch_bbox_gts, num_classes)
        print("train ap: ", ap)
        
        # validation
        with torch.no_grad():
            epoch_detections = []
            epoch_cls_gts = []
            epoch_bbox_gts = []
            for data in tqdm(val_dataloader):
                imgs, cls_targets, bbox_targets = data
                imgs = imgs.cuda()
                cls_targets = cls_targets.cuda()
                bbox_targets = bbox_targets.cuda()
                optimizer.zero_grad()
                cls_preds, center_preds, reg_preds = model(imgs)
                # cls_loss, center_loss, bbox_loss = model.calc_loss(cls_preds, center_preds, reg_preds, 
                #             bbox_targets, cls_targets, anchor_points)
                # loss = model.weighted_sum_loss(cls_loss, center_loss, bbox_loss)

                detections = model.postprocess(cls_preds, center_preds, reg_preds, anchor_points, 640, 416)
                for i in range(train_dataloader.batch_size):
                    img_cls_targets = cls_targets[i]
                    img_bbox_targets = bbox_targets[i]
                    pos_id = torch.nonzero(img_cls_targets != num_classes)
                    # print(img_cls_targets[pos_id].shape, img_bbox_targets[pos_id].shape)
                    epoch_detections.append(detections[i])
                    epoch_cls_gts.append(img_cls_targets[pos_id].reshape(-1))
                    epoch_bbox_gts.append(img_bbox_targets[pos_id].reshape(-1, 4))
            
            ap = eval(epoch_detections, epoch_cls_gts, epoch_bbox_gts, num_classes)
            print("val ap: ", ap)
        
        if num_batch >= opt['warmup_steps']:
            lr_scheduler.step()
