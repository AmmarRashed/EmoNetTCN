import torch
import torch.nn as nn
import torchvision

from TCN import TemporalConvNet


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ResTCN(nn.Module):
    def __init__(self):
        super(ResTCN, self).__init__()

        self.spatial_feat_dim = 32
        self.num_classes = 4
        self.nhid = 128
        self.levels = 8
        self.kernel_size = 7
        self.dropout = .1
        self.channel_sizes = [self.nhid] * self.levels

        self.tcn = TemporalConvNet(
            self.spatial_feat_dim,
            self.channel_sizes,
            kernel_size=self.kernel_size,
            dropout=self.dropout)
        self.linear = nn.Linear(self.channel_sizes[-1], 1)
        self.sigmoid = nn.Sigmoid()

        num_ftrs = 512

    def forward(self, embedding):
        z = embedding.transpose(1, 2)
        y = self.tcn(z)
        output = self.sigmoid(self.linear(torch.sum(y, dim=2)))

        return output