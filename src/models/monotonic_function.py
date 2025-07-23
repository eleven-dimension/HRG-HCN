import torch
import torch.nn as nn
import torch.nn.functional as F


# trainable parameter: a := [a_0, a_1, ..., a_{k}]
# \Delta_d = softplus(a_d) + \epsilon
# r_d = \sum_{i = 0}^d \Delta_d
class MonotonicFunction(nn.Module):
    def __init__(self, k, epsilon=1e-6):
        super().__init__()
        self.k = k
        self.epsilon = epsilon
        self.a = nn.Parameter(torch.randn(k + 1))

    def forward(self):
        deltas = F.softplus(self.a) + self.epsilon
        r = torch.cumsum(deltas, dim=0)
        return r


if __name__ == "__main__":
    v = MonotonicFunction(k=5)
    v.eval()
    with torch.no_grad():
        print(v().detach().cpu().numpy())
