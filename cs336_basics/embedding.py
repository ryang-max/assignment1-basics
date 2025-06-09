import torch
import einops

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = torch.nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    
    def update_weight(self, new_weight: torch.Tensor):
        assert new_weight.shape == (self.num_embeddings, self.embedding_dim)
        with torch.no_grad():
            self.weight = torch.nn.Parameter(new_weight)