import torch
import torch.nn.functional as F
import numpy as np
import cv2

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, cls_labels, bbox_labels):
        for t in self.transforms:
            img, cls_labels, bbox_labels = t(img, cls_labels, bbox_labels)
        return img, cls_labels, bbox_labels

class Resize(object):
    def __init__(self, size, keep_ratio=True):
        self.size = size
        self.keep_ratio = keep_ratio
    
    def __call__(self, img, cls_labels, bbox_labels):
        src_h, src_w = img.shape[:2]
        dst_w, dst_h = self.size
        scale_w = dst_w / src_w
        scale_h = dst_h / src_h
        if self.keep_ratio:
            if scale_w < scale_h:
                tmp_h = int(src_h * scale_w)
                img = cv2.resize(img, (dst_w, tmp_h))
                pad_img = np.zeros((dst_h, dst_w, 3), dtype=img.dtype)
                pad_img[:tmp_h, :dst_w] = img
                bbox_labels = bbox_labels * scale_w
            else:
                tmp_w = int(src_w * scale_h)
                img = cv2.resize(img, (tmp_w, dst_h))
                pad_img = np.zeros((dst_h, dst_w, 3), dtype=img.dtype)
                pad_img[:dst_h, :tmp_w] = img
                bbox_labels = bbox_labels * scale_h
            return pad_img, cls_labels, bbox_labels
        else:
            img = cv2.resize(img, self.size)
            bbox_labels[:, 0::2] = bbox_labels[:, 0::2] * scale_w
            bbox_labels[:, 1::2] = bbox_labels[:, 1::2] * scale_h
            return img, cls_labels, bbox_labels

class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, cls_labels, bbox_labels):
        """
        Args:
            imgs: (h, w, 3)
            cls_labels: (n)
            bbox_labels: (n, 4)
        """
        if torch.rand(1) < self.prob:
            w = img.shape[1]
            img = img[:, ::-1]
            bbox_labels[:, 0::2] = w - bbox_labels[:, 2::-2]
        return img, cls_labels, bbox_labels

class Pad(object):
    def __init__(self, pad=10):
        self.pad = pad

    def __call__(self, img, cls_labels, bbox_labels):
        return img, cls_labels, bbox_labels

class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, img, cls_labels, bbox_labels):
        img = img / 255.0
        img = (img - self.mean) / self.std
        return img, cls_labels, bbox_labels

class ToTorchTensor(object):
    def __init__(self):
        pass

    def __call__(self, img, cls_labels, bbox_labels):
        img = torch.from_numpy(img).permute(2, 0, 1)
        cls_labels = torch.from_numpy(cls_labels)
        bbox_labels = torch.from_numpy(bbox_labels)
        return img, cls_labels, bbox_labels