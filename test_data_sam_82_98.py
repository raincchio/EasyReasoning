#!/usr/bin/env python
# coding: utf-8

# In[18]:


from dataclasses import dataclass
from typing import Tuple, Any, Dict, Sequence
import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from datasets import load_dataset
from torch.utils.data import DataLoader
import random

import copy

from tqdm import trange

torch._dynamo.config.compiled_autograd = True
torch.set_float32_matmul_precision('high')

# [Set hyperparams here]
@dataclass
class HRMConfig:
    vocab_size: int = 10  # Sudoku digits 0(unfilled) .. 9
    seq_len: int = 82  # Sudoku has 9x9 = 81 cells + BOS

    hidden_size: int = 256
    intermediate_size: int = 256
    batch_size: int = 256
    head_dim: int = 64
    is_causal: bool = False

    num_layers: int = 4

    H_cycles: int = 2
    L_cycles: int = 8

    cycle_per_data: int = 16

    norm_eps: float = 1e-6
    rope_base: float = 10000.0
    forward_dtype: str = "bfloat16" # change to float32 if your hardware doesn't support bfloat16

    seed: int = 7

@dataclass
class TrainConfig:

    epochs: int = 5
    cycle_per_data: int = 16

    lr: float = 0.001
    weight_decay: float = 0.1

    ema:bool = True
    ema_rate: float = 0.999


# In[19]:


def set_up(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    torch.set_num_threads(1)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# In[20]:


# borrow code from https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/main/models/ema.py
class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}
        self.tmp = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.tmp[name] = param.data.clone()
                param.data.copy_(self.shadow[name].data)

    def normal(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.tmp[name].data)

# In[21]:


# [Model implementation]

CosSin = Tuple[torch.Tensor, torch.Tensor]

def trunc_normal_init_(x: torch.Tensor, std: float):
    return nn.init.trunc_normal_(x, std=std).mul_(1)

def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x: torch.Tensor, cos_sin: CosSin):
    # q, k: [..., seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    cos, sin = cos_sin
    return ((x * cos.unsqueeze(-2)) + (rotate_half(x) * sin.unsqueeze(-2))).to(x.dtype)

class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool, batch_output_dims: Sequence[int] = (), **kwargs):
        super().__init__()
        self.in_features = in_features

        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((*batch_output_dims, out_features, in_features), **kwargs), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features, ), **kwargs))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.view(-1, self.in_features).to(input.dtype), self.bias.to(input.dtype) if self.bias is not None else None)

class CastedScaledEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Scale to the same std as most parameters
        self.scale = embedding_dim ** 0.5
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=1.0 / self.scale)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.scale * self.weight.to(self.cast_to))

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached

class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, **kwargs):
        super().__init__()
        self.gate_up_proj = CastedLinear(hidden_size, intermediate_size, bias=False, batch_output_dims=(2, ), **kwargs)
        self.down_proj = CastedLinear(intermediate_size, hidden_size, bias=False, **kwargs)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, is_causal, **kwargs):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.is_causal = is_causal

        self.qkv_proj = CastedLinear(hidden_size, self.num_heads * self.head_dim, bias=False, batch_output_dims=(3, ), **kwargs)
        self.o_proj = CastedLinear(head_dim * num_heads, hidden_size, bias=False, **kwargs)
        with torch.no_grad():
            self.o_proj.weight.zero_()

    def forward(self, hidden_states: torch.Tensor, cos_sin: CosSin) -> torch.Tensor:
        # hidden_states, qkv: [..., seq_len, hidden_size]
        qkv = self.qkv_proj(hidden_states)

        # Split head (last dimension of projected qkv)
        qkv = rearrange(qkv, "... (h hd) -> ... h hd", h=self.num_heads)
        query, key, value = qkv.chunk(3, dim=-1)
        # Rotary embedding
        query = apply_rotary_pos_emb(query, cos_sin)
        key = apply_rotary_pos_emb(key, cos_sin)
        # PyTorch SDPA attention
        # query, key, value: [... x seq_len x num_heads x head_dim]
        attn_output = F.scaled_dot_product_attention(query.transpose(-2, -3), key.transpose(-2, -3), value.transpose(-2, -3), is_causal=self.is_causal).transpose(-2, -3)
        # attn_output: [..., seq_len, num_heads, head_dim]
        attn_output = rearrange(attn_output, "... h hd -> ... (h hd)")
        return self.o_proj(attn_output)

class TransformerBlock(nn.Module):
    def __init__(self, config: HRMConfig) -> None:
        super().__init__()
        self.attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_heads=config.hidden_size // config.head_dim,
            is_causal=config.is_causal
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size
        )
        self.norm = lambda x: F.rms_norm(x, (x.shape[-1], ), eps=config.norm_eps)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:  # Post Norm
        x = self.norm(x + self.attn(x, **kwargs))
        return self.norm(x + self.mlp(x))

class HRMRecurrentBlock(nn.Module):
    def __init__(self, config: HRMConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(config) for _layer_idx in range(config.num_layers)])

    def forward(self, x: torch.Tensor, n: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x + n
        for layer in self.layers:
            h = layer(h, **kwargs)
        return h

# HRMCarry is a tuple containing two latent states(z_H, z_L)
HRMCarry = Tuple[torch.Tensor, torch.Tensor]
datatype ={
    'float32':torch.float32,
    'bfloat16':torch.bfloat16,
}

class HRM(nn.Module):
    def __init__(self, config: HRMConfig) -> None:
        super().__init__()
        self.H_cycles = config.H_cycles
        self.L_cycles = config.L_cycles

        self.hidden_size = config.hidden_size
        self.seq_len = config.seq_len
        self.dtype =datatype[config.forward_dtype]

        self.batch_size = config.batch_size

        # Backbone Layers
        self.H_level = HRMRecurrentBlock(config)
        self.L_level = HRMRecurrentBlock(config)
        self.register_buffer('carry_h', trunc_normal_init_(
            torch.empty(config.batch_size, self.seq_len, self.hidden_size, dtype=self.dtype), std=1.0))
        self.register_buffer('carry_l', trunc_normal_init_(
            torch.empty(config.batch_size, self.seq_len, self.hidden_size, dtype=self.dtype), std=1.0))

        # RoPE
        self.rope = RotaryEmbedding(config.head_dim, config.seq_len, config.rope_base)
        # I/O Layers
        self.embed = CastedScaledEmbedding(config.vocab_size, config.hidden_size, cast_to=self.dtype)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)

        self._carry_l = trunc_normal_init_(
            torch.empty(config.batch_size, self.seq_len, self.hidden_size, dtype=self.dtype), std=1.0)
        self._carry_h = trunc_normal_init_(
            torch.empty(config.batch_size, self.seq_len, self.hidden_size, dtype=self.dtype), std=1.0)

    def keep_carry(self, ):
        self._carry_h.copy_(self.carry_h)
        self._carry_l.copy_(self.carry_l)

    def restore_carry(self, ):
        self.carry_h.copy_(self._carry_h)
        self.carry_l.copy_(self._carry_l)

    def init_carry(self, ):
        self.carry_h = trunc_normal_init_(
            torch.empty(self.batch_size, self.seq_len, self.hidden_size, dtype=self.dtype), std=1.0)
        self.carry_l = trunc_normal_init_(
            torch.empty(self.batch_size, self.seq_len, self.hidden_size, dtype=self.dtype), std=1.0)


    def forward(self, input_ids: torch.Tensor):
        x = self.embed(input_ids)
        seq_info = dict(cos_sin=self.rope())
        # Forward iterations
        with torch.no_grad():
            # compatible for inference
            z_H = self.carry_h[:input_ids.shape[0]]
            z_L = self.carry_l[:input_ids.shape[0]] # Unpack tuple
            for _i in range(self.H_cycles - 1):
                for _j in range(self.L_cycles):
                    z_L = self.L_level(z_L, z_H + x, **seq_info)
                z_H = self.H_level(z_H, z_L, **seq_info)

        for _j in range(self.L_cycles):
            z_L = self.L_level(z_L, z_H+x, **seq_info)

        z_H = self.H_level(z_H, z_L , **seq_info)

        self.carry_h = torch.cat((z_H, self.carry_h[input_ids.shape[0]:]), dim=0).detach()
        self.carry_l = torch.cat((z_L, self.carry_l[input_ids.shape[0]:]), dim=0).detach()

        return self.lm_head(z_H)  # Return tuple and ensure no gradient moves across carry


# In[22]:


# [Training and Inference Step]
# @torch.compile(dynamic=False)
def train_step(model: nn.Module, opt: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor):

    y_hat = model(x)
    # loss (f32 for CrossEntropy)
    # params = {k: v.clone() for k, v in model.H_level.state_dict().items()}
    loss = F.cross_entropy(y_hat.view(-1, y_hat.shape[-1]).to(torch.float32), y.view(-1), reduction="mean")
    loss.backward()
    opt.step()
    opt.zero_grad()


    # metrics
    with torch.no_grad():
        preds = torch.argmax(y_hat, dim=-1)
        metrics = {
            "loss": loss.detach(),
            "per_position_accuracy": torch.mean(preds == y, dtype=torch.float32),
            "exact_match": torch.mean(torch.all(preds == y, dim=-1), dtype=torch.float32)
        }

    return metrics

# [Dataloader and training loop]
def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    # Create a random digit mapping: a permutation of 1..9, with zero (blank) unchanged
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))

    # Randomly decide whether to transpose.
    transpose_flag = np.random.rand() < 0.5

    # Generate a valid row permutation:
    # - Shuffle the 3 bands (each band = 3 rows) and for each band, shuffle its 3 rows.
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

    # Similarly for columns (stacks).
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    # Build an 81->81 mapping. For each new cell at (i, j)
    # (row index = i // 9, col index = i % 9),
    # its value comes from old row = row_perm[i//9] and old col = col_perm[i%9].
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        # Apply transpose flag
        if transpose_flag:
            x = x.T
        # Apply the position mapping.
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        # Apply digit mapping
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)

def collate_fn(batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for item in batch:
        board = np.frombuffer(item["question"].replace('.', '0').encode(), dtype=np.uint8).reshape(9, 9) - ord('0')
        solution = np.frombuffer(item["answer"].encode(), dtype=np.uint8).reshape(9, 9) - ord('0')
        # Convert and flatten
        board = board.flatten().astype(np.int32)
        solution = solution.flatten().astype(np.int32)
        # Pad a BOS token
        xs.append(np.pad(board, (1, 0)))
        ys.append(np.pad(solution, (1, 0)))

    return torch.from_numpy(np.stack(xs, axis=0)), torch.from_numpy(np.stack(ys, axis=0))

def create_dataloader(split: str, batch_size: int):

    data_files = {
        'train':'data/train.csv',
        'test_hard':'data/test_hard.csv',
        'test_sudoku_bench':'data/test_sudoku_bench.csv',
        'train_aug': 'data/train_aug.csv',
    }
    dataset = load_dataset('csv', data_files=data_files, split=split)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=len(dataset) >= batch_size,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues in Jupyter
        prefetch_factor=None,
        persistent_workers=False  # Must be False when num_workers=0
    )


# In[ ]:


# traing model
model_config, train_config = HRMConfig(), TrainConfig()
set_up(model_config.seed)
device = torch.accelerator.current_accelerator(check_available=True)
if device is None:
    device = torch.device("cpu")

print (f"Training on {device.type}")

# Initialize
train_loader = create_dataloader("train_aug", model_config.batch_size)
total_steps = int(train_config.cycle_per_data * len(train_loader) * train_config.epochs)

with torch.device(device):
    model = HRM(model_config)
    # model = torch.compile(model, dynamic=False, fullgraph=True)

from muon import SingleDeviceMuon

opt = SingleDeviceMuon(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)


# Train & Eval loop
ema_helper = None
if train_config.ema:
    print('Setup EMA')
    ema_helper = EMAHelper(mu=train_config.ema_rate)
    ema_helper.register(model)


eval_loaders = {split_name: create_dataloader(split_name, model_config.batch_size) for split_name in ["test_hard"]}

import datetime, os
timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
current_file_name = os.path.basename(__file__)
log_file= open(f'{timestamp}_{current_file_name[:-3]}_log.txt', 'w')

for epoch in range(1, train_config.epochs+1):
    model.train()
    step = 0
    for x, y in train_loader:
        x = x.to(device)
        y = y.long().to(device)

        for cycle in range(train_config.cycle_per_data):
            metrics = train_step(model, opt, x, y)
            if train_config.ema:
                ema_helper.update(model)
            # if metrics['exact_match']>0.99:
            #     break
        step += 1
        # scheduler.step()

        if step%25==0:
            info = f"Ep {epoch}/{train_config.epochs} Step {step}/{len(train_loader)}: " + ', '.join(f'{k}={v.item() if isinstance(v, torch.Tensor) else v:.3f}' for k, v in metrics.items())
            print (info)
            log_file.write(info+'\n')

    model.eval()
    if train_config.ema:
        print("SWITCH TO EMA")
        ema_helper.ema(model)

    model.keep_carry()
    for eval_name, eval_loader in eval_loaders.items():
        num_total = 0

        num_correct = 0
        for x, y in eval_loader:
            for cycle in range(model_config.cycle_per_data):
                y_hat = model(x.to(device))
            y_hat = torch.argmax(y_hat, dim=-1)
            num_total += y.shape[0]
            num_correct += torch.all(y_hat == y.to(device), dim=-1).sum().item()
            model.restore_carry()
        info =  f"[Eval Set {eval_name}]" + f"Solved: {100 * num_correct / num_total:.2f}% ({num_correct}/{num_total})"
        print(info)
        log_file.write(info + '\n')

    # if train_config.ema:
    #     print("SWITCH TO Normal")
    #     ema_helper.normal(model)

    # torch.save(model.state_dict(), f"model/HRM_Mini_ema_{epoch}.pth")
    # print(f'model saved at epoch {epoch}, location: model/HRM_Mini_ema_{epoch}.pth')

