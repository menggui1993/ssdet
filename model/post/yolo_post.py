import torch
import torch.nn as nn

class Yolov3Post(nn.Module):
    """
    yolov3 post process and loss calculation
    """
    def __init__(self,
                 num_classes,
                 anchors):
        super(Yolov3Post, self).__init__()