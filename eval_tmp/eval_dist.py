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

from torch.utils.data import DataLoader
import random

# torch._dynamo.config.compiled_autograd = True
torch.set_float32_matmul_precision('high')

from util.encoding import SudokuEncoder, RelationNetwork
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
# [Set hyperparams here]
@dataclass
class HRMConfig:
    vocab_size: int = 11  # Sudoku digits 0(unfilled) .. 9
    seq_len: int = 81  # Sudoku has 9x9 = 81 cells + BOS

    hidden_size: int = 256
    # intermediate_size: int = 256
    batch_size: int = 2048
    num_heads: int = 8
    is_causal: bool = False
    expansion:int =  4

    num_layers: int = 4

    H_layers: int = 4
    L_layers: int = 4

    H_cycles: int = 2
    L_cycles: int = 2

    puzzle_emb_ndim:int = 256
    num_puzzle_identifiers: int = 1
    pos_encodings:str = 'learned'

    cycle_per_data: int = 16

    # norm_eps: float = 1e-4
    rope_base: float = 10000.0
    forward_dtype: str = "float32" # change to float32 if your hardware doesn't support bfloat16

    seed: int = 42

    epochs: int = 200

    lr: float = 3e-5
    weight_decay: float = 0

    ema:bool = False
    ema_rate: float = 1 - 1e-5
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    def to_dict(self):
        return {
            "num_layers": self.num_layers, # depth
            "H_cycles": self.H_cycles, # H
            "L_cycles": self.L_cycles, # R
            "cycle_per_data": self.cycle_per_data, # Rec
            "hidden_size":self.hidden_size # width
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

import os
def train():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    # world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    model_config  = tyro.cli(HRMConfig)
    set_up(model_config.seed)

    from hrm import HierarchicalRecurrentModelV1
    model = HierarchicalRecurrentModelV1(model_config).cuda(local_rank)
    state_dict = torch.load('step_166300')
    if any(k.startswith('_orig_mod.model.') for k in state_dict.keys()):
        state_dict = {k[len('_orig_mod.model.'):] if k.startswith('_orig_mod.model.') else k: v
                      for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    # model = torch.compile(model, dynamic=False, fullgraph=True)
    model = DDP(model, device_ids=[local_rank],static_graph=True)

    from torch.utils.data.distributed import DistributedSampler
    from util.data import SudokuminiDataset
    f_str = 'train'

    eval_dataset = SudokuminiDataset(f'data/{f_str}_convert.csv')
    eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
    eval_loader= DataLoader(eval_dataset, batch_size=model_config.batch_size, sampler=eval_sampler, num_workers=4, pin_memory=True)
    # eval_loader = DataLoader(eval_dataset, batch_size=model_config.batch_size, shuffle=False, num_workers=4,
    #                          pin_memory=True)
    model.eval()

    fail_x = []
    fail_y = []
    if local_rank==0:
        from tqdm import tqdm
        eval_loader = tqdm(eval_loader)
    with torch.no_grad():
        for x, y in eval_loader:
            init_hidden = True
            puzzle_ids = torch.zeros((x.shape[0], 1), dtype=torch.long).cuda(local_rank)
            for cycle in range(model_config.cycle_per_data):
                y_hat = model(x.cuda(local_rank), puzzle_identifiers=puzzle_ids,init_hidden=init_hidden)
                init_hidden = False
            y_hat = torch.argmax(y_hat, dim=-1)-1
            # answer = (torch.argmax(logits, dim=-1) - 1)
            idx = ((y.cuda(local_rank) == y_hat).sum(-1) != 81).nonzero().cpu().flatten()

            len_idx = len(idx)
            if local_rank==0:
                print(len_idx)

            for i in range(len_idx):
                qst_ = x[idx[i]]-1
                ans_ = y[idx[i]]
                qst_num = qst_ + ord('0')  # 1->'1', 2->'2' ... '9'->'9'
                qst_str = qst_num.numpy().astype('uint8').tobytes().decode()
                fail_x.append(qst_str)

                ans_num = ans_ + ord('0')  # 1->'1', 2->'2' ... '9'->'9'
                ans_str = ans_num.numpy().astype('uint8').tobytes().decode()
                fail_y.append(ans_str)
            # break
    data_dict = {'question':fail_x, 'answer':fail_y}
    import pandas as pd
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(f'fail/{f_str}_{local_rank}.csv', index=False)

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__=='__main__':

    train()

