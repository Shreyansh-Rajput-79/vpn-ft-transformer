import math

import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """Turns each numeric feature into its own token: token_j = x_j * W_j + b_j.

    Every feature has its own weight and bias, so the transformer knows which
    column each token came from.
    """

    def __init__(self, num_features, d_model):
        super().__init__()
        bound = 1 / math.sqrt(d_model)
        self.weight = nn.Parameter(torch.empty(num_features, d_model).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(num_features, d_model).uniform_(-bound, bound))

    def forward(self, x):
        return x.unsqueeze(-1) * self.weight + self.bias


class FTTransformer(nn.Module):
    def __init__(self, num_features, num_classes, d_model=64, n_heads=4,
                 n_layers=3, dim_feedforward=128, dropout=0.1):
        super().__init__()

        self.tokenizer = FeatureTokenizer(num_features, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes),
        )

    def _tokens(self, x):
        x = self.tokenizer(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        return torch.cat([cls, x], dim=1)

    def forward(self, x):
        x = self.transformer(self._tokens(x))
        return self.head(x[:, 0, :])

    @torch.no_grad()
    def forward_with_attention(self, x):
        """Same output as forward() (in eval mode), plus the attention maps.

        Returns (logits, attn) where attn is [batch, layers, tokens, tokens],
        averaged over heads. Token 0 is CLS, token j+1 is feature j.
        """
        h = self._tokens(x)
        maps = []
        for layer in self.transformer.layers:  # pre-norm encoder layer, unrolled
            z = layer.norm1(h)
            out, weights = layer.self_attn(z, z, z, need_weights=True, average_attn_weights=True)
            h = h + layer.dropout1(out)
            h = h + layer._ff_block(layer.norm2(h))
            maps.append(weights)
        return self.head(h[:, 0, :]), torch.stack(maps, dim=1)
