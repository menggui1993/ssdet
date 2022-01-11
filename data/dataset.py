import torch
from torch.utils import data
import os
from pathlib import Path
import cv2
import numpy as np
from data.transforms import Compose, Resize, Normalize, RandomHorizontalFlip, ToTorchTensor

class YoloDataset(data.Dataset):
    def __init__(self, img_dir, label_dir, classes, max_objs=50, transforms=None):
        self.classes = classes
        self.imgs = []
        self.labels = []
        self.max_objs = max_objs
        if transforms is None:
            self.transforms = Compose([Resize(size=(640,640)),
                                       Normalize(),
                                       ToTorchTensor()])
        else:
            self.transforms = transforms

        num_classes = len(classes)
        for _, _, files in os.walk(img_dir):
            for img_file in files:
                if (Path(img_file).suffix not in ['.jpeg','.jpg','.png','.bmp']):
                    continue
                label_file = Path(img_file).stem + '.txt'
                label_file = os.path.join(label_dir, label_file)
                if not os.path.exists(label_file):
                    print("Can't find annotation file ", label_file)
                    continue
                img_file = os.path.join(img_dir, img_file)
                self.imgs.append(os.path.join(img_dir, img_file))
                cls_labels, bbox_labels = self.readAnnotation(label_file, img_file)
                # pad labels with empty gt so that data can be batched
                gts = cls_labels.shape[0]
                pad_cls_labels = np.ones((max_objs,), dtype=cls_labels.dtype)*num_classes
                pad_bbox_labels = np.zeros((max_objs,4), dtype=bbox_labels.dtype)
                if gts > max_objs:
                    pad_cls_labels = cls_labels[:max_objs]
                    pad_bbox_labels = bbox_labels[:max_objs, :]
                elif gts > 0:
                    pad_cls_labels[:gts] = cls_labels
                    pad_bbox_labels[:gts, :] = bbox_labels

                self.labels.append((pad_cls_labels, pad_bbox_labels))

    def __getitem__(self, index):
        img = cv2.imread(self.imgs[index])
        cls_labels, bbox_labels = self.labels[index]
        img, cls_labels, bbox_labels = self.transforms(img, cls_labels, bbox_labels)
        return img, cls_labels, bbox_labels

    def __len__(self):
        return len(self.imgs)

    def get_num_classes(self):
        return len(self.classes)

    def readAnnotation(self, label_file, img_file):
        cls_labels = []
        bbox_labels = []
        img = cv2.imread(img_file)
        img_h, img_w = img.shape[:2]
        with open(label_file, 'rt') as f:
            while True:
                obj_data = f.readline()
                if not obj_data:
                    break
                obj_data = obj_data.split(' ')
                cls_labels.append(int(obj_data[0]))
                ct_x = float(obj_data[1])
                ct_y = float(obj_data[2])
                w = float(obj_data[3])
                h = float(obj_data[4])
                left = (ct_x - w/2) * img_w
                right = (ct_x + w/2) * img_w
                top = (ct_y - h/2) * img_h
                bot = (ct_y + h/2) * img_h
                bbox_labels.append([left, top, right, bot])
        cls_labels = np.array(cls_labels, dtype=np.int64)
        bbox_labels = np.array(bbox_labels, dtype=np.float64)
        return cls_labels, bbox_labels

if __name__ == '__main__':
    img_dir = "/home/twc/ddisk/fabric_defect/20211101_data/fabric12_yolo/val/images"
    label_dir = "/home/twc/ddisk/fabric_defect/20211101_data/fabric12_yolo/val/labels"
    classes = ("BB", "CS", "DJ", "DPD", "DW", "GS", "JK", "PD", "WZ", "ZH", "ZK", "ZW")
    dataset = YoloDataset(img_dir, label_dir, classes)

    dataloader = data.DataLoader(dataset, 4)
    for imgs, cls_labels, bbox_labels in dataloader:
        # print(type(imgs))
        # print(type(cls_labels))
        print(imgs.shape)
        print(cls_labels.shape)
        # print(bbox_labels.shape)