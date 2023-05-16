import torch
from torch import nn

from TCN import TemporalConvNet


class ResTCN(nn.Module):
    def __init__(self):
        super(ResTCN, self).__init__()

        self.tcn = TemporalConvNet(
            32,
            [128] * 8,
            kernel_size=7,
            dropout=.1).cuda()

        self.fc = nn.Linear(512, 32).cuda()
        self.linear = nn.Linear(128, 1).cuda()

    def forward(self, x):
        x = self.fc(x)
        x = x.transpose(1, 2)
        y = self.tcn(x)
        output = self.linear(torch.sum(y, dim=2))
        return output.squeeze(1)
