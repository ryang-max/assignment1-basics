import torch
import einops

class SiLU(torch.nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
class GLU(torch.nn.Module):
    def __init__(self, device=None, dtype=None, sigma=torch.nn.Sigmoid):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.activation = sigma()

    def forward(self, x: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor) -> torch.Tensor:
        gate = self.activation(einops.einsum(
            W1, x, 
            "d_ff d_model, ... d_model -> ... d_ff"
        ))
        var = einops.einsum(
            W2, x, 
            "d_ff d_model, ... d_model -> ... d_ff"
        )
        return torch.mul(gate, var)
    
class SwiGLU(torch.nn.Module):
    def __init__(self, d_ff: int, d_model: int, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.d_ff = d_ff
        self.d_model = d_model
        self.glu = GLU(device=device, dtype=dtype, sigma=torch.nn.SiLU)

    def update_weights(self, W1: torch.Tensor, W2: torch.Tensor, W3: torch.Tensor):
        assert W1.shape == (self.d_ff, self.d_model), "W1 shape mismatch"
        assert W2.shape == (self.d_model, self.d_ff), "W2 shape mismatch"
        assert W3.shape == (self.d_ff, self.d_model), "W3 shape mismatch"
        self.W1 = W1
        self.W2 = W2
        self.W3 = W3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(
            self.W2 , self.glu(x, self.W1, self.W3),
            "d_model d_ff, ... d_ff -> ... d_model"
        )