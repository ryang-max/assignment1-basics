import torch
import einops

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.dim = d_k
        self.max_seq_len = max_seq_len
        self.device = device if device is not None else torch.device('cpu')
        self._build_cache()
        
    def _build_cache(self):
        sqe_ids = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.device)
        thetas = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=self.device) / self.dim))
        freqs = torch.outer(sqe_ids, thetas)
        self.register_buffer('cosine_cached', freqs.cos(), persistent=False)
        self.register_buffer('sine_cached', freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.dim, "Input tensor's last dimension must match the embedding dimension."
        assert token_positions.shape[-1] <= self.max_seq_len, "Token positions exceed maximum sequence length."
        assert x.shape[-2] == token_positions.shape[-1], "Input tensor's second last dimension must match token positions length."

        seq_len = token_positions.shape[-1]
        cosine = self.cosine_cached[token_positions]
        sine = self.sine_cached[token_positions]
        #print(f"cosine shape: {cosine.shape}, sine shape: {sine.shape}, x shape: {x.shape}")
        cosine = cosine.unsqueeze(0)
        sine = sine.unsqueeze(0)
        #print(f"cosine unsqueezed shape: {cosine.shape}, sine unsqueezed shape: {sine.shape}")
        x_even = x[..., 0::2]
        x_ord = x[..., 1::2]
        output1 = x_even * cosine - x_ord * sine
        output2 = x_even * sine + x_ord * cosine
        return torch.flatten(
            torch.stack((output1, output2), dim=-1),
            start_dim=-2
        )