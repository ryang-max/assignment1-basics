import torch
import einops

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.dim = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.gain = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        norm = x / rms * self.gain
        return norm.to(in_type)

    def update_gain(self, new_gain: torch.Tensor):
        assert new_gain.shape == (self.dim,)
        with torch.no_grad():
            self.gain = torch.nn.Parameter(new_gain)
    