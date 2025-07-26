import numpy as np
import torch

from layers import MonotonicFunction
from configs import MonotonicFunctionCfg


def test_monotonic_function():
    cfg = MonotonicFunctionCfg(5)
    monotonic_layer = MonotonicFunction(cfg)
    monotonic_layer.eval()
    with torch.no_grad():
        arr = monotonic_layer().detach().cpu().numpy()
    assert len(arr) == 6, f"Expected length 6, got {len(arr)}"
    assert np.all(np.diff(arr) > 0), "Output is not strictly increasing"
