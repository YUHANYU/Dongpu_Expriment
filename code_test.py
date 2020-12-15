"""
代码测试
"""

import torch
import torch.nn as nn
from torch import optim, autograd

input = torch.FloatTensor([[0.1, 0.3],
                           [0.4, 0.5],
                           [0.8, 0.1],
                           [0.9, 0.3]])
target = torch.LongTensor([1, 0, 0, 0])
criterion = nn.functional.cross_entropy(input, target)
print(criterion)
