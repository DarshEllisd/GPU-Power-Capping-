import torch
import torch.nn as nn
class Model(nn.Module):
    """
    Elementwise addition.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        return x + y
batch_size = 4096
dim = 39321


def get_inputs():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.rand(batch_size, dim, device=device)
    y = torch.rand(batch_size, dim, device=device)
    return [x, y]

def get_init_inputs():
    return []
