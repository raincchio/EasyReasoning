import torch
import torch.nn as nn


class SudokuEncoder(nn.Module):
    def __init__(self, digit_dim=16, pos_dim=8):
        super().__init__()

        # 0-9 (0 表示空)
        self.digit_emb = nn.Embedding(10, digit_dim)

        # row / col
        self.row_emb = nn.Embedding(9, pos_dim)
        self.col_emb = nn.Embedding(9, pos_dim)

        self.out_dim = digit_dim + 2 * pos_dim

        # 预生成 row col index
        rows = []
        cols = []

        for i in range(81):
            rows.append(i // 9)
            cols.append(i % 9)

        self.register_buffer("row_index", torch.tensor(rows))
        self.register_buffer("col_index", torch.tensor(cols))

    def forward(self, x):
        """
        x: (batch, 81)  每个位置 0-9
        """

        digit = self.digit_emb(x)                 # (B,81,16)
        row = self.row_emb(self.row_index)        # (81,8)
        col = self.col_emb(self.col_index)        # (81,8)

        pos = torch.cat([row, col], dim=-1)       # (81,16)
        pos = pos.unsqueeze(0).expand(x.size(0), -1, -1)

        out = torch.cat([digit, pos], dim=-1)     # (B,81,32)

        return out      # (B,81*32)

# import torch
# import torch.nn as nn

class RelationNetwork(nn.Module):
    def __init__(self, d_model, hidden):
        super().__init__()

        self.g = nn.Sequential(
            nn.Linear(2*d_model, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )

        self.f = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )

    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape

        xi = x.unsqueeze(2).expand(B, N, N, D)
        xj = x.unsqueeze(1).expand(B, N, N, D)

        pair = torch.cat([xi, xj], dim=-1)   # [B,N,N,2D]

        r = self.g(pair)                     # [B,N,N,H]

        r = r.sum(dim=(1,2))                 # [B,H]

        relation = self.f(r)

        relation = relation.unsqueeze(1)  # [B, 1, H]
        relation = relation.expand(B, N, D)

        return relation+x