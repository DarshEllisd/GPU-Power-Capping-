import torch
import torch.nn as nn
class Model(nn.Module):
    """
    Deep fully connected network.
    """
    def __init__(self):
        super(Model, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(2048, 2048) for _ in range(6)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x
batch_size = 4096
dim = 3932
def get_inputs():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.rand(batch_size, 2048, device=device)
    return [x]

def get_init_inputs():
    return []
