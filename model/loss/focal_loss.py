import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=True, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        num_pos = (targets.reshape(-1) != 0).nonzero().shape[0]

        pt = torch.exp(-BCE_loss)
        F_loss = (self.alpha*targets + (1-self.alpha)*(1-targets)) * (1-pt).pow(self.gamma) * BCE_loss

        if self.reduction == "mean":
            F_loss = F_loss.sum() / num_pos
        elif self.reduction == "sum":
            F_loss = F_loss.sum()
        
        return F_loss

if __name__ == '__main__':
    fl = FocalLoss(0.7, 2)
    
    num_anchors = 8
    num_classes = 4
    batch_size = 16

    inputs = torch.randn(batch_size, num_anchors, num_classes)
    targets = torch.randint(0, num_classes, (batch_size, num_anchors))
    targets = F.one_hot(targets).type_as(inputs)
    
    loss = fl(inputs, targets)
    print(loss)