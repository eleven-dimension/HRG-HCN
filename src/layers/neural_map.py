import torch.nn as nn
import torch.nn.functional as F


class NeuralMap(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralMap, self).__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, v):
        h = F.relu(self.W1(v))
        return self.W2(h)
