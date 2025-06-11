import torch
import einops

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_sub = x - x.amax(dim=dim, keepdim=True)
    exp_sum = x_sub.exp().sum(dim=dim, keepdim=True)
    return x_sub.exp() / exp_sum

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    d_k = Q.shape[-1]

    scores = einops.einsum(
        Q, K,
        "... queries d_k, ... keys d_k -> ... queries keys"
    ) / (d_k ** 0.5)
    
    if mask is not None:
        scores = mask * scores
        scores = scores.masked_fill(scores == 0, float("-inf"))

    scores = softmax(scores)

    return einops.einsum(
        scores, V,
        "... queries keys, ... keys d_v-> ... queries d_v"
    )
    
class CausalMultiHeadAttention(torch.nn.Module):

    def __init__(self, d_model: int, num_heads: int):
        self.d_model = d_model
        self.num_heads = num_heads
        super().__init__()
        self.d_k = d_model // num_heads

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, O: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        qkv_proj = torch.stack((Q, K, V), dim=0)
        qkv_proj = einops.rearrange(
            qkv_proj, "qkv (h d) d_in -> qkv h d d_in",
            h=self.num_heads, d=self.d_k
        )
        qkv_proj = einops.einsum(
            qkv_proj, x,
            "qkv h d d_in, ... seq d_in -> qkv h seq d"
        )
        q_proj, k_proj, v_proj = qkv_proj

        mask = torch.triu(
            torch.ones(x.shape[-2], x.shape[-2]),
        )
        mask = torch.stack(
            [mask] * self.num_heads, dim=0
        )
        print(f"q_proj shape: {q_proj.shape}, k_proj shape: {k_proj.shape}, v_proj shape: {v_proj.shape}, mask shape: {mask.shape}")

        scores = scaled_dot_product_attention(
            Q=q_proj, K=k_proj, V=v_proj, mask=mask
        )
        scores = einops.rearrange(
            scores, "h d_k d_v -> (h d_k) d_v"
        )
        output = O @ scores
        return output

        


