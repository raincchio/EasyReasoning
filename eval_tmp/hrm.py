import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

def trunc_normal_init_(tensor, std=1.0, lower=torch.tensor(-2.0), upper=torch.tensor(2.0)):
    """Truncated normal initialization"""
    # lower = torch.tensor(lower, device=tensor.device)
    # upper = torch.tensor(upper, device=tensor.device)
    with torch.no_grad():
        if std == 0: return tensor.zero_()
        sqrt2 = torch.sqrt(torch.tensor(2.0))
        a, b = torch.erf(lower/sqrt2), torch.erf(upper/sqrt2)
        z, c = (b-a)/2, (2*torch.pi)**-0.5
        pdf_u, pdf_l = c*torch.exp(-0.5*lower**2), c*torch.exp(-0.5*upper**2)
        comp_std = std / torch.sqrt(1 - (upper*pdf_u - lower*pdf_l)/z - ((pdf_u-pdf_l)/z)**2)
        tensor.uniform_(a, b).erfinv_().mul_(sqrt2*comp_std).clip_(lower*comp_std, upper*comp_std)
    return tensor

def _find_multiple(a, b): return (-(a // -b)) * b

def rotate_half(x):
    mid = x.shape[-1] // 2
    return torch.cat((-x[..., mid:], x[..., :mid]), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    orig_dtype = q.dtype
    q, k = q.to(cos.dtype), k.to(cos.dtype)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)

def rms_norm(x, eps):
    dtype = x.dtype
    x = x.float()
    return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)).to(dtype)


class CastedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.weight = nn.Parameter(trunc_normal_init_(torch.empty(out_features, in_features), std=1.0/(in_features**0.5)))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        weight = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)

class CastedEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, init_std, cast_to):
        super().__init__()
        self.cast_to = cast_to
        self.embedding_weight = nn.Parameter(trunc_normal_init_(torch.empty(num_embeddings, embedding_dim), std=init_std))

    def forward(self, x):
        return F.embedding(x, self.embedding_weight.to(self.cast_to))

class CastedSparseEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, batch_size, init_std, cast_to):
        super().__init__()
        self.cast_to = cast_to
        self.weights = nn.Buffer(trunc_normal_init_(torch.empty(num_embeddings, embedding_dim), std=init_std), persistent=True)
        self.local_weights = nn.Buffer(torch.zeros(batch_size, embedding_dim, requires_grad=True), persistent=False)
        self.local_ids = nn.Buffer(torch.zeros(batch_size, dtype=torch.int32), persistent=False)

    def forward(self, inputs):
        if not self.training: return self.weights[inputs].to(self.cast_to)
        with torch.no_grad():
            self.local_weights.copy_(self.weights[inputs])
            self.local_ids.copy_(inputs)
        return self.local_weights.to(self.cast_to)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self): return self.cos_cached, self.sin_cached

class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()
        self.hidden_size, self.head_dim, self.num_heads = hidden_size, head_dim, num_heads
        self.num_key_value_heads, self.causal = num_key_value_heads, causal
        self.output_size = head_dim * num_heads
        self.qkv_proj = CastedLinear(hidden_size, (num_heads + 2*num_key_value_heads) * head_dim)
        self.o_proj = CastedLinear(self.output_size, hidden_size)

    def forward(self, cos_sin, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states).view(batch_size, seq_len, self.num_heads + 2*self.num_key_value_heads, self.head_dim).transpose(-2, -3)
        query, key, value = qkv[:, :self.num_heads], qkv[:, self.num_heads:self.num_heads+self.num_key_value_heads], qkv[:, self.num_heads+self.num_key_value_heads:]

        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        attn_output = F.scaled_dot_product_attention(query=query, key=key, value=value, is_causal=self.causal)
        return self.o_proj(attn_output.transpose(-2, -3).view(batch_size, seq_len, self.output_size))

class SwiGLU(nn.Module):
    def __init__(self, hidden_size, expansion):
        super().__init__()
        intermediate_size = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.gate_proj = CastedLinear(hidden_size, intermediate_size)
        self.up_proj = CastedLinear(hidden_size, intermediate_size)
        self.down_proj = CastedLinear(intermediate_size, hidden_size)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class HierarchicalRecurrentModelV1Config(BaseModel):
    batch_size: int; seq_len: int; puzzle_emb_ndim: int = 0; num_puzzle_identifiers: int
    vocab_size: int; H_cycles: int; L_cycles: int; H_layers: int; L_layers: int
    hidden_size: int; expansion: float; num_heads: int; pos_encodings: str
    rms_norm_eps: float = 1e-5; rope_theta: float = 10000.0; forward_dtype: str = "float32"

class HierarchicalRecurrentModelV1Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config.hidden_size, config.hidden_size//config.num_heads, config.num_heads, config.num_heads)
        self.mlp = SwiGLU(config.hidden_size, config.expansion)
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin, hidden_states):
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin, hidden_states), self.norm_eps)
        return rms_norm(hidden_states + self.mlp(hidden_states), self.norm_eps)

class HierarchicalRecurrentModelV1RecurrentModule(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states, input_injection, **kwargs):
        hidden_states = hidden_states + input_injection
        for layer in self.layers: hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states

class HierarchicalRecurrentModelV1(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        self.config = config_dict
        self.forward_dtype = torch.float32
        self.embed_scale = torch.sqrt(torch.tensor(256))
        embed_init_std = 1.0 / self.embed_scale

        # Core components
        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, embed_init_std, self.forward_dtype)
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size)

        # Puzzle embedding
        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    self.config.batch_size, 0, self.forward_dtype)

        # Position encodings
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(self.config.hidden_size//self.config.num_heads,
                                              self.config.seq_len + self.puzzle_emb_len, self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, embed_init_std, self.forward_dtype)

        # HRM layers

        self.H_level = HierarchicalRecurrentModelV1RecurrentModule([HierarchicalRecurrentModelV1Block(self.config) for _ in range(self.config.H_layers)])
        self.L_level = HierarchicalRecurrentModelV1RecurrentModule([HierarchicalRecurrentModelV1Block(self.config) for _ in range(self.config.L_layers)])
        self.z_H_0 = nn.Buffer(trunc_normal_init_(torch.empty(256, dtype=torch.float32), std=1), persistent=True).unsqueeze(0).unsqueeze(0).expand(1, 82, -1)
        self.z_L_0 = nn.Buffer(trunc_normal_init_(torch.empty(256, dtype=torch.float32), std=1), persistent=True).unsqueeze(0).unsqueeze(0).expand(1, 82, -1)

        self.z_H=None
        self.z_L=None
        # self._init_hidden()

    def _init_hidden(self):
        self.z_H=self.z_H_0.clone().cuda()
        self.z_L=self.z_L_0.clone().cuda()

    def _input_embeddings(self, input, puzzle_identifiers):
        embedding = self.embed_tokens(input.to(torch.int32))

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        return self.embed_scale * embedding

    def forward(self,inputs, puzzle_identifiers, init_hidden):
        if init_hidden:
            self._init_hidden()
        z_H = self.z_H
        z_L = self.z_L
        seq_info = dict(cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None)
        input_embeddings = self._input_embeddings(inputs, puzzle_identifiers)

        with torch.no_grad():
            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                if _H_step != self.config.H_cycles - 1:
                    z_H = self.H_level(z_H, z_L, **seq_info)

        # Final steps
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        self.z_H = z_H
        self.z_L = z_L

        logits = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        return logits