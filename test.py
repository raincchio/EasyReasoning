import torch
import torch.nn as nn

# 输入的形状 (batch_size, sequence_length, channels)
x = torch.randn(32, 100, 3)  # 32个样本，每个样本100个时间步，每个时间步有3个通道

conv = nn.Conv1d(in_channels=100, out_channels=16, kernel_size=3)
output = conv(x)
print(x.shape)