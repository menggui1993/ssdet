import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    """
    Mish activation function
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * torch.tanh(F.softplus(x))
        return x
        

if __name__ == '__main__':
    mish = Mish()
    x = torch.rand(5)
    y = mish(x)
    print(y)