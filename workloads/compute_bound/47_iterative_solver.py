import torch
import torch.nn as nn
class Model(nn.Module):
    """
    Jacobi-style iterative solver.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        for _ in range(20):
            x = (x + torch.roll(x, 1, 0) + torch.roll(x, -1, 0) +
                 torch.roll(x, 1, 1) + torch.roll(x, -1, 1)) / 5
        return x

def get_inputs():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.rand(4096, 4096, device=device)
    return [x]

def get_init_inputs():
    return []
