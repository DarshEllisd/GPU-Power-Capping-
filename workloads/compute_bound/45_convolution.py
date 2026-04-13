import torch
import torch.nn as nn
class Model(nn.Module):
    """
    CNN convolution.
    """
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(128, 128, 3, padding=1)

    def forward(self, x):
        return self.conv(x)

def get_inputs():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.rand(64, 128, 256, 256, device=device)
    return [x]

def get_init_inputs():
    return []
