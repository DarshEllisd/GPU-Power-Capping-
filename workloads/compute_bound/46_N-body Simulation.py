import  torch
import torch.nn as nn
class Model(nn.Module):
    """
    N-body simulation.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, pos):
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        dist = torch.norm(diff, dim=2) + 1e-5
        force = diff / dist.unsqueeze(2)**3
        return force.sum(dim=1)

def get_inputs():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pos = torch.rand(2000, 3, device=device)
    return [pos]

def get_init_inputs():
    return []
