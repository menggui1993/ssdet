import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_module.conv_block import ConvBlock

# ref: https://github.com/miguelvr/dropblock
class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)

class SPP(nn.Module):
    def __init__(self,
                 in_ch,
                 kernel_size = (5, 9, 13),
                 act_cfg = dict(type='Relu', args=dict())):
        super(SPP, self).__init__()
        self.poolings = []
        for kernel in kernel_size:
            self.add_module("pooling_" + str(kernel), nn.MaxPool2d(kernel, stride=1, padding=kernel//2))
            self.poolings.append("pooling_"+str(kernel))
        self.conv = ConvBlock(in_ch*4, in_ch, 1, act_cfg=act_cfg)

    def forward(self, x):
        outs = [x]
        for pooling_layer in self.poolings:
            pooling = getattr(self, pooling_layer)
            outs.append(pooling(x))
        out = torch.cat(outs, dim=1)
        out = self.conv(out)
        return out