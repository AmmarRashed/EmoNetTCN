import torch
import torch.nn as nn


class ImageEmbeddingRegressor(nn.Module):
    def __init__(self, d_model=512, num_layers=2, num_heads=1, dropout=0.1, seq_len=100):
        super(ImageEmbeddingRegressor, self).__init__()

        # Encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        # pooling
        self.pooling = nn.AdaptiveMaxPool1d(1)
        # decoder
        self.decoder = nn.Linear(d_model, 1)

    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # x is a tensor of shape (batch_size, seq_len, d_model)
        x = x.transpose(0, 1)  # shape: (seq_len, batch_size, d_model)

        # Apply Transformer Encoder
        x = self.transformer_encoder(x)  # [T, N, H]
        x = x.permute(1, 2, 0)  # [N, H, T]
        x = self.pooling(x)
        x = x.squeeze(-1)
        # apply decoder
        output = torch.sigmoid(self.decoder(x)).squeeze(-1)
        # Apply sigmoid activation to output to constrain it between 0 and 1
        output = torch.sigmoid(output)

        return output
