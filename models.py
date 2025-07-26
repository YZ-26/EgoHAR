import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
import torch.nn.functional as F


class FeatureEncoder(nn.Module):
    def __init__(self):
        super(FeatureEncoder, self).__init__()

        # Load pretrained VGG16 and keep only convolutional layers
        weights = EfficientNet_B3_Weights.DEFAULT
        model = efficientnet_b3(weights=weights)
        self.features = model.features  # Output shape: (B, 1536, 7, 7) if input is 224x224

        # Attention prediction layer
        self.attention_conv = nn.Conv2d(in_channels=1536, out_channels=1, kernel_size=1)

        # Reduce feature channels from 1536 to 100
        self.channel_reduction = nn.Sequential(
            nn.Conv2d(1536, 100, kernel_size=1),
            nn.BatchNorm2d(100),
            nn.ReLU()
        )

    def forward(self, x):  # x: (B, 3, H, W)
        features = self.features(x)  # (B, 1536, H', W')

        # Predict attention map
        attention_map = self.attention_conv(features)  # (B, 1, H', W')
        attention_map = torch.sigmoid(attention_map)  # constrain to [0,1]

        # Broadcast attention over channels and apply
        attended_features = features * attention_map  # (B, 1536, H', W')

        # Residual connection
        out = features + attended_features

        # Reduce channels
        out = self.channel_reduction(out)  # (B, 6, H', W')

        return out, attention_map

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=16): # max_len: 16 which is frame sequence length
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]  # (B, T, D)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.5):
        super(TransformerEncoderLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        x_ff = self.linear2(F.gelu(self.linear1(x)))
        x = self.norm2(x + x_ff)
        return x

# class TransformerEncoderLayer(nn.Module):
#     def __init__(self, embed_dim, num_heads, dense_dim, dropout=0.2):
#         super(TransformerEncoderLayer, self).__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads

#         self.norm_input = nn.LayerNorm(embed_dim)  # LayerNorm before attention
#         self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

#         self.norm_1 = nn.LayerNorm(embed_dim)
#         self.norm_2 = nn.LayerNorm(embed_dim)

#         self.ff = nn.Sequential(
#             nn.Linear(embed_dim, dense_dim),
#             nn.GELU(),
#             nn.BatchNorm1d(dense_dim),
#             nn.Dropout(dropout),
#             nn.Linear(dense_dim, embed_dim),
#             nn.BatchNorm1d(embed_dim),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x, mask=None):
#         # x: (B, T, D)
#         x = self.norm_input(x)

#         attn_output, _ = self.attn(x, x, x)

#         x = self.norm_1(x + attn_output)

#         # Feedforward with BatchNorm1d: need shape (B*T, D)
#         B, T, D = x.shape
#         proj_input = x
#         proj_output = self.ff(x.view(B * T, D)).view(B, T, D)

#         return self.norm_2(proj_input + proj_output)


class TransformerClassifier(nn.Module):
    def __init__(self, num_classes, input_dim=4900, embed_dim=512, num_heads=4, hidden_dim=100, num_layers=1): # embed_dim: From EffNet output, 6x7x7
        super(TransformerClassifier, self).__init__()
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.layers = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
                          nn.Linear(embed_dim, 256),
                          nn.ReLU(),
                          # nn.Dropout(0.5),
                          # nn.Linear(128, 64),
                          # nn.ReLU(),
                          # nn.Dropout(0.5),
                          nn.Linear(256, num_classes)
                      )
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),  # 4900 → 512 or whatever you define
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):  # x: (B, T, D)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(1, 2)  # (B, D, T) for pooling
        x = self.pool(x).squeeze(-1)  # (B, D)
        return self.classifier(x)