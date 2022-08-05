import torch.nn as nn
from mmcv.runner.base_module import BaseModule


class TCGap(nn.Module):
    def __init__(self, input_lv=-1):
        super().__init__()
        self.input_lv = input_lv

    def forward(self, x):
        return x[self.input_lv]['x'].mean(dim=1)
