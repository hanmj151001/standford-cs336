# cs336_basics/model.py
import torch 
import torch.nn as nn 
from cs336_basics.layers import Linear, Embedding, RMSNorm, MultiHeadAttention, MLP


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: float, max_seq_len: int, device = None):
        super().__init__()
        
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        self.device = device if device else torch.device("cpu")
        
        
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer("inv_freq", inv_freq)
        
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(*x.shape[:-1], -1, 2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim = -1)
        
        return x.flatten(start_dim=-2)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:

        seq_len = x.size(-2)
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
            token_positions = token_positions.unsqueeze(0).expand(x.size(0), -1)

        # (B, S), (d_k // 2) => (B, S, d_k // 2)
        # Equal to token_positions[..., :, None] * self.inv_freq[None, None, :]
        theta = torch.einsum("... n, d ->  ... n d", token_positions, self.inv_freq)
        
        # (B, S, d_k // 2) => (B, S, d_k)
        cos = theta.cos().repeat_interleave(2, dim=-1)
        sin = theta.sin().repeat_interleave(2, dim=-1)
        
        x = x * cos + self._rotate_half(x) * sin

        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        use_rope: bool = False,
        max_seq_len: int = 512,
        theta: float = 10000.0,
    ):
        super().__init__()
        
        self.mha = MultiHeadAttention(
            d_model, num_heads, use_rope, max_seq_len, theta
        )
        
        self.ffn = MLP(
            d_model,
            d_ff,
        )
        
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mha(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
    
class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        """
        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            num_heads (int): Number of heads to use in multi-headed attention.
            d_ff (int): Dimensionality of the feedforward layer.
            num_layers (int): Number of transformer layers.
            use_rope (bool): Whether to use RoPE.
            max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
            theta (float): RoPE parameter.
        """
        super().__init__()
        self.embedding = Embedding(
            num_embedding=vocab_size, 
            embedding_dim=d_model,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    use_rope=True,
                    max_seq_len=context_length,
                    theta=rope_theta,
                )
                for _ in range(num_layers)
            ]
        )

        self.rms_norm = RMSNorm(d_model=d_model)
        self.output_embeddings = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x)

        x = self.rms_norm(x)
        x = self.output_embeddings(x)

        return x