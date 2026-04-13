import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Split and recombine tensor.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        chunks = torch.chunk(x, 4, dim=1)
        return torch.cat(chunks, dim=1)

batch_size = 4096
dim = 39321


def get_inputs():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.rand(batch_size, dim, device=device)
    return [x]

def get_init_inputs():
    return []
