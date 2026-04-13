import torch
import torch.nn as nn
class Model(nn.Module):
    """
    Scatter write workload.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, idx, val):
        x[idx] = val
        return x
    
batch_size = 4096*2
dim = 39321

def get_inputs():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    size = batch_size * dim
    x = torch.zeros(size, device=device)
    idx = torch.randint(0, size, (size,), device=device)
    val = torch.rand(size, device=device)
    return [x, idx, val]

def get_init_inputs():
    return []

