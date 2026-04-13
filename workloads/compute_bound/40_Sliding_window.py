import torch
import torch.nn as nn   
class Model(nn.Module):
    """
    Sliding window stencil.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return x[:-1, :-1] + x[1:, :-1] + x[:-1, 1:] + x[1:, 1:]

def get_inputs():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.rand(40960, 4096, device=device)
    return [x]

def get_init_inputs():
    return []
