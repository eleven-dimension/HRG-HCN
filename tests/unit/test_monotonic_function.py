import numpy as np
import torch

from layers import MonotonicFunction


def test_monotonic_function():
    monotonic_layer = MonotonicFunction(5)
    monotonic_layer.eval()
    with torch.no_grad():
        arr = monotonic_layer().detach().cpu().numpy()
    assert len(arr) == 6, f"Expected length 6, got {len(arr)}"
    assert np.all(np.diff(arr) > 0), "Output is not strictly increasing"
