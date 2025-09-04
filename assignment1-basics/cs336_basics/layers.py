# cs336_basics/layers.py
import torch 
import torch.nn as nn
import math
from cs336_basics.model import RotaryPositionalEmbedding

class Linear(nn.Module):
    # def___init__(self, in_features: int, out_features: int, device = None, dtype = None):
    def __init__(
        self, in_features: int, out_features: int, device= None, dtype = None 
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
    
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
            
        std = math.sqrt((2 / (self.in_features + self.out_features)))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std) # 截断正态分布初始化
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, device = None, dtype = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )

        a = math.sqrt(3.0) / math.sqrt(self.embedding_dim)  # 均匀分布初始化
        nn.init.uniform_(self.weight, -a, +a)

    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(
        self, normalized_shape: int, eps: float = 1e-8, device = None, dtype = None
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(normalized_shape, device=device, dtype=dtype)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        norm_x = torch.rsqrt(self.eps + 
                             torch.mean(x ** 2, dim = -1, keepdim=True)
        )
        return x * norm_x * self.weight
    
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x = x - torch.max(x, dim=dim, keepdim=True).values
    x = torch.exp(x)
    x = x / torch.sum(x, dim=dim, keepdim=True)
    return x

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    d_k = q.size(-1)
    
    scores = torch.matual(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attn_weights = softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output
    
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        max_seq_len: int | None = None,
        theta: float | None = None,
        token_positions: torch.Tensor | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.head_dim = d_model // num_heads

        self.use_rope = use_rope
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.token_positions = token_positions

        if use_rope and (max_seq_len is not None and theta is not None):
            self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        
        self.out_proj = Linear(d_model, d_model)
    
    def _causal_mask(self, seq_len: int) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        mask =  mask.unsqueeze(0).unsqueeze(0)
        return mask

    def forward(self, in_features: torch.Tensor):
        """
        in_features: (B, S, D)
        """
        B, S, D = in_features.size()
        
        q = self.q_proj(in_features).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(in_features).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(in_features).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q = self.rope(q, self.token_positions)
            k = self.rope(k, self.token_positions)
        
        mask = self._causal_mask(S) 
        mask = mask.to(q.device)
        
        attn_output = scaled_dot_product_attention(q, k, v, mask)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)
        output = self.out_proj(attn_output)
        return output
    

def SwiGLU(x: torch.Tensot) -> torch.Tensor:
    return x * torch.sigmoid(x)
    
class MLP(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, device=None, dtype=None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        output = self.w2(SwiGLU(self.w1(x)) * self.w3(x)) # 区别于普通的MLP, W_down*Relu(W_up*x)
        return output