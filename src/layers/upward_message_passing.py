from typing import Any
import torch
from torch_geometric.nn import MessagePassing


class DownwardMessagePassing(MessagePassing):
    def __init__(self):
        super().__init__()

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return super().message(x_j)

    def aggregate(
        self,
        inputs: torch.Tensor,
        index: torch.Tensor,
        ptr: torch.Tensor | None = None,
        dim_size: int | None = None,
    ) -> torch.Tensor:
        return super().aggregate(inputs, index, ptr, dim_size)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return super().forward(*args, **kwargs)
