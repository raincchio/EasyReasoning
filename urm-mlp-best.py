#!/usr/bin/env python
# coding: utf-8

# In[18]:
import tyro

from dataclasses import dataclass
from typing import Tuple, Any, Dict, Sequence
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from datasets import load_dataset
from torch.utils.data import DataLoader
import random
import torch.nn.init as init
# torch._dynamo.config.compiled_autograd = True
torch.set_float32_matmul_precision('high')

from util.encoding import SudokuEncoder, RelationNetwork

# [Set hyperparams here]
@dataclass
class HRMConfig:
    vocab_size: int = 10  # Sudoku digits 0(unfilled) .. 9
    seq_len: int = 81  # Sudoku has 9x9 = 81 cells + BOS

    hidden_size: int = 512
    # intermediate_size: int = 256
    batch_size: int = 1024
    head_dim: int = 64
    is_causal: bool = False

    num_layers: int = 8

    H_cycles: int = 1
    L_cycles: int = 1

    cycle_per_data: int = 16

    norm_eps: float = 1e-4
    rope_base: float = 10000.0
    forward_dtype: str = "float32" # change to float32 if your hardware doesn't support bfloat16

    seed: int = 42

    epochs: int = 5

    lr: float = 3e-5
    weight_decay: float = 0

    ema:bool = False
    ema_rate: float = 1 - 3e-5

    def to_dict(self):
        return {
            "num_layers": self.num_layers, # depth
            "H_cycles": self.H_cycles, # H
            "L_cycles": self.L_cycles, # R
            "cycle_per_data": self.cycle_per_data, # Rec
            "hidden_size":self.hidden_size, # width,
            'ema':self.ema
        }

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

class MLPSeqBlock(nn.Module):
    def __init__(self, config: HRMConfig, **kwargs) -> None:
        super().__init__()

        self.in_proj = CastedLinear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            bias=False,
            **kwargs
        )

        self.rel_proj = CastedLinear(
            in_features=config.seq_len,
            out_features=config.seq_len,
            bias=False,
            **kwargs
        )

        self.out_proj = CastedLinear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            bias=False,
            **kwargs
        )

        self.norm = lambda x: F.rms_norm(x, (x.shape[-1],), eps=config.norm_eps)
        self.act = F.silu # activation function. U can try anything like ReLU, SiLU, ...

    def forward(self, x:torch.Tensor, **kwargs):
        in_proj = self.norm(self.act(self.in_proj(x)) + x).transpose(-1, -2)
        rel_proj = self.norm(self.act(self.rel_proj(in_proj)) + in_proj).transpose(-1, -2)
        return self.norm(self.act(self.out_proj(rel_proj)) + rel_proj)

class GluBlock(nn.Module):
    def __init__(self, config: HRMConfig) -> None:
        super().__init__()

        self.mlp = ConvSwiGLU(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size
        )
        self.norm = lambda x: F.rms_norm(x, (x.shape[-1], ), eps=config.norm_eps)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:  # Post Norm
        # x = self.norm(x + self.attn(x, **kwargs))
        return self.norm(x + self.mlp(x))

class HRMRecurrentBlock(nn.Module):
    def __init__(self, config: HRMConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([MLPSeqBlock(config) for _layer_idx in range(config.num_layers)])

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

        self.sudoku_encoder = SudokuEncoder(digit_dim=config.hidden_size//2, pos_dim=config.hidden_size//4)

        # Backbone Layers
        self.H_level = HRMRecurrentBlock(config)
        self.L_level = HRMRecurrentBlock(config)
        self.register_buffer('carry_h', trunc_normal_init_(
            torch.empty(config.batch_size, self.seq_len, self.hidden_size, dtype=self.dtype), std=1.0))
        self.register_buffer('carry_l', trunc_normal_init_(
            torch.empty(config.batch_size, self.seq_len, self.hidden_size, dtype=self.dtype), std=1.0))

        # RoPE
        # self.rope = RotaryEmbedding(config.head_dim, config.seq_len, config.rope_base)
        # I/O Layers
        self.embed = CastedScaledEmbedding(config.vocab_size, config.hidden_size, cast_to=self.dtype)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)

        self._carry_l = trunc_normal_init_(
            torch.empty(config.batch_size, self.seq_len, self.hidden_size, dtype=self.dtype), std=1.0)
        self._carry_h = trunc_normal_init_(
            torch.empty(config.batch_size, self.seq_len, self.hidden_size, dtype=self.dtype), std=1.0)

    def keep_carry(self, ):
        self._carry_h.copy_(self.carry_h.data)
        self._carry_l.copy_(self.carry_l.data)

        # self._carry_h, self.carry_h

    def restore_carry(self, ):
        self.carry_h.copy_(self._carry_h.data)
        self.carry_l.copy_(self._carry_l.data)

    def init_carry(self, ):
        # self.carry_h = trunc_normal_init_(
        #     torch.empty(self.batch_size, self.seq_len, self.hidden_size, dtype=self.dtype), std=1.0)
        self.carry_l = trunc_normal_init_(
            torch.empty(self.batch_size, self.seq_len, self.hidden_size, dtype=self.dtype), std=1.0)


    def forward(self, input_ids: torch.Tensor):
        # x = self.embed(input_ids)
        x = self.sudoku_encoder(input_ids)
        seq_info = dict(cos_sin=self.rope())
        # Forward iterations
        with torch.no_grad():
            # compatible for inference
            z_H = self.carry_h[:input_ids.shape[0]]
            z_L = self.carry_l[:input_ids.shape[0]] # Unpack tuplev

        z_L = self.L_level(z_L, z_H + x, **seq_info)



        self.carry_h = torch.cat((z_H, self.carry_h[input_ids.shape[0]:]), dim=0).detach()
        self.carry_l = torch.cat((z_L, self.carry_l[input_ids.shape[0]:]), dim=0).detach()
        if model_config.H_cycles!=0:
            return self.lm_head(z_H)  # Return tuple and ensure no gradient moves across carry
        else:
            return self.lm_head(z_L)


# In[22]:


# [Training and Inference Step]
@torch.compile(dynamic=False)
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


# In[ ]:



model_config  = tyro.cli(HRMConfig)
set_up(model_config.seed)
device = torch.accelerator.current_accelerator(check_available=True)
if device is None:
    device = torch.device("cpu")

print (f"Training on {device.type}")

# Initialize
data_name = ['train','test']

from util.data import SudokuDataset

dataset = SudokuDataset('data/train_convert.csv')

train_loader = DataLoader(dataset, batch_size=model_config.batch_size, shuffle=True, num_workers=4,
                          pin_memory=True)

eval_dataset = SudokuDataset('data/test_convert.csv')

eval_loader = DataLoader(eval_dataset, batch_size=model_config.batch_size, shuffle=True, num_workers=4,
                         pin_memory=True)

total_steps = int(model_config.cycle_per_data * len(train_loader) * model_config.epochs)

with torch.device(device):
    model = HRM(model_config)
    model = torch.compile(model, dynamic=False, fullgraph=True)

# from muon import SingleDeviceMuon
from torch.optim import Adam

opt = Adam(model.parameters(), lr=model_config.lr, weight_decay=model_config.weight_decay)


# Train & Eval loop
ema_helper = None
if model_config.ema:
    print('Setup EMA')
    ema_helper = EMAHelper(mu=model_config.ema_rate)
    ema_helper.register(model)


import datetime, os
timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
current_file_name = os.path.basename(__file__)
wanted_dict = model_config.to_dict()
exp_str = "-".join(f"{k}_{v}" for k,v in wanted_dict.items())

log_file= open(f'{exp_str}.txt', 'w')
print(model_config)
log_file.write(str(model_config))
log_file.flush()

for epoch in range(1, model_config.epochs+1):
    model.train()
    step = 0
    model.keep_carry()
    for x, y in train_loader:
        x = x.to(device)
        y = y.long().to(device)

        for cycle in range(model_config.cycle_per_data):
            metrics = train_step(model, opt, x, y)
            if model_config.ema:
                ema_helper.update(model)

        step += 1

        model.restore_carry()
        if step%25==0:
            info = f"Ep {epoch}/{model_config.epochs} Step {step}/{len(train_loader)}: " + ', '.join(f'{k}={v.item() if isinstance(v, torch.Tensor) else v:.6f}' for k, v in metrics.items())
            print (info)
            log_file.write(info+'\n')
            log_file.flush()

    model.eval()
    if model_config.ema:
        print("SWITCH TO EMA")
        ema_helper.ema(model)

    num_total = 0
    num_correct = 0
    for x, y in eval_loader:
        for cycle in range(model_config.cycle_per_data):
            y_hat = model(x.to(device))
        y_hat = torch.argmax(y_hat, dim=-1)
        num_total += y.shape[0]
        num_correct += torch.all(y_hat == y.to(device), dim=-1).sum().item()
        model.restore_carry()
    info =  f"[Eval Set test ]" + f"Solved: {100 * num_correct / num_total:.2f}% ({num_correct}/{num_total})"
    print(info)
    log_file.write(info + '\n')
    log_file.flush()

