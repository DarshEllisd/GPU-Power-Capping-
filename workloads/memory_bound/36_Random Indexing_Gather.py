import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Random gather workload (memory-bound, non-coalesced access).
    """
    def __init__(self, repeats=50):
        super(Model, self).__init__()
        self.repeats = repeats

    def forward(self, a, idx):
        for _ in range(self.repeats):
            b = a[idx]   # random memory access
        return b

size = 1_000_000

def get_inputs():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    a = torch.randn(size, device=device)
    idx = torch.randint(0, size, (size,), device=device)
    return [a, idx]

def get_init_inputs():
    return []

