import torch
import einops


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_dim = in_features
        self.out_dim = out_features
        self.device = device
        self.dtype = dtype
        self.W = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.W)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.in_dim
        return einops.einsum(
            self.W, x, 
            "out_dim in_dim, ... in_dim -> ... out_dim"
        )
    
    def update_weight(self, new_weight: torch.Tensor):
        assert new_weight.shape == (self.out_dim, self.in_dim)
        with torch.no_grad():
            self.W = torch.nn.Parameter(new_weight)
    