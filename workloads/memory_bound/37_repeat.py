import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Repeat tensor along dimension.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return x.repeat(1, 2)

batch_size = 4096*2
dim = 39321

def get_inputs():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.rand(batch_size, dim, device=device)
    return [x]

def get_init_inputs():
    return []
