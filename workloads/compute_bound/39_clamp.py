import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Clamp operation (low compute, memory-bound).
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return torch.clamp(x, -0.5, 0.5)

batch_size = 4096
dim = 3932

def get_inputs():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.rand(batch_size, dim, device=device)
    return [x]

def get_init_inputs():
    return []
