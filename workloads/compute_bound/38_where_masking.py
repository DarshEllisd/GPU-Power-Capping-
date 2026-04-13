import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Conditional selection using mask.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, mask):
        return torch.where(mask, x, y)

batch_size = 4096
dim = 3932

def get_inputs():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.rand(batch_size, dim, device=device)
    y = torch.rand(batch_size, dim, device=device)
    mask = torch.rand(batch_size, dim, device=device) > 0.5
    return [x, y, mask]

def get_init_inputs():
    return []
