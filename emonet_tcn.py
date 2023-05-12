import torch
import torch.nn as nn

from tcn import TCN


class EmoNetFeatureExtractor(nn.Module):
    def __init__(self, emo_net):
        super(EmoNetFeatureExtractor, self).__init__()
        self.feature_extractor = nn.Sequential(*list(emo_net.children())[:-1])

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class EmoNetTCN(nn.Module):
    def __init__(self, emo_net):
        super(EmoNetTCN, self).__init__()
        self.emo_net = EmoNetFeatureExtractor(emo_net)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.spatial_feat_dim = 32
        self.nhid = 128
        self.dropout = 0.1
        self.kernel_size = 7
        self.levels = 8
        self.channel_sizes = [self.nhid] * self.levels

        self.tcn = TCN(
            self.spatial_feat_dim,
            self.channel_sizes,
            kernel_size=self.kernel_size,
            dropout=self.dropout)

        self.eng_out = nn.Linear(self.channel_sizes[-1], 4)  # TODO change output to 1 to make it for regression

    def forward(self, x):
        z = torch.zeros([x.shape[0], x.shape[1], self.spatial_feat_dim]).to(self.device)
        for t in range(x.size(1)):
            x = self.model_conv(x[:, t, :, :, :])
            z[:, t, :] = x

        z = z.transpose(1, 2)
        y = self.tcn(z)
        eng_output = self.eng_out(torch.sum(y, dim=2))

        return eng_output
