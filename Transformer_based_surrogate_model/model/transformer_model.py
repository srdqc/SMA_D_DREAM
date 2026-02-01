import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding module.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register positional encoding as a buffer (not a trainable parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to the input tensor.

        Parameters:
        x (Tensor): Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
        Tensor: Position-encoded tensor
        """
        x = x + self.pe[:x.size(1), :]
        return x


class CustomTransformer(nn.Module):
    def __init__(self, input_len=99, output_len=71, d_model=512, nhead=8, num_layers=6):
        super().__init__()

        # Input embedding layer
        self.embedding = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(d_model, nhead)
        self.encoder = TransformerEncoder(encoder_layers, num_layers)

        # Sequence length adapter (Method 1: linear projection)
        self.seq_adapter = nn.Linear(input_len * d_model, output_len * d_model)

        # Output projection layer
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        Forward pass of the custom Transformer model.

        Parameters:
        x (Tensor): Input tensor of shape [batch_size, 99]

        Returns:
        Tensor: Output tensor of shape [batch_size, 71]
        """
        # Expand input dimension
        x = x.unsqueeze(-1)              # [batch_size, 99, 1]
        x = self.embedding(x)             # [batch_size, 99, d_model]
        x = self.pos_encoder(x)

        # Transformer encoder processing
        x = x.permute(1, 0, 2)            # Transformer expects [seq_len, batch, dim]
        x = self.encoder(x)
        x = x.permute(1, 0, 2)            # Back to [batch, seq_len, dim]

        # Adjust sequence length
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)     # [batch_size, 99 * d_model]
        x = self.seq_adapter(x)            # [batch_size, 71 * d_model]
        x = x.reshape(batch_size, 71, -1)  # [batch_size, 71, d_model]

        # Output projection
        x = self.output_layer(x)           # [batch_size, 71, 1]
        return x.squeeze(-1)               # [batch_size, 71]


# Example usage
model = CustomTransformer()
input = torch.randn(32, 99)  # Assume batch_size = 32
output = model(input)        # Output shape: [32, 71]
