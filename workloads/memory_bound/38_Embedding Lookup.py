import torch
import torch.nn as nn
class Model(nn.Module):
    """
    Embedding lookup (memory-heavy).
    """
    def __init__(self, vocab_size=1_000_000, dim=128):
        super(Model, self).__init__()
        self.emb = nn.Embedding(vocab_size, dim)

    def forward(self, idx):
        return self.emb(idx)

batch_size = 6553


def get_inputs():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    idx = torch.randint(0, 1_000_000, (batch_size,), device=device)
    return [idx]

def get_init_inputs():
    return []

