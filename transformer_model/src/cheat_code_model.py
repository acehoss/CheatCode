import torch
import torch.nn as nn
import torch.nn.functional as F
import math  # Add this import for positional encoding

# Add the PositionalEncoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1440):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# Define the Ticker Subnetwork simplified for single time scale
class TickerSubNetwork(nn.Module):
    def __init__(self, d_model, nhead, num_layers, seq_len, dropout=0.1, dim_feedforward=2048):
        super(TickerSubNetwork, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                layer_norm_eps=1e-4,
                activation=nn.Mish()
            ),
            num_layers=num_layers,
        )
        self.input_projection = nn.Linear(15, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout, max_len=seq_len)
        self.seq_len = seq_len
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            # For Transformer layers, we can rely on their default initialization

    def forward(self, data_minute):
        # data_minute shape: (batch_size, seq_len, num_features)
        x = self.input_projection(data_minute)  # Project input features to d_model
        x = self.positional_encoding(x)         # Apply positional encoding
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)  # Aggregate across time (batch_first=True)
        return x

# CheatCodeModel class with additional transformer layers after cross-attention
class CheatCodeModel(nn.Module):
    def __init__(
        self,
        num_tickers,
        d_model_input,
        nhead_input,
        num_encoder_layers,
        seq_len,
        dropout=0.1,
        num_post_layers=2,
        d_model_post=None,
        nhead_post=None,
        dim_feedforward_input=2048,
        dim_feedforward_post=None,  # Changed default to None
    ):
        super(CheatCodeModel, self).__init__()
        self.num_tickers = num_tickers
        
        # Set dimensions and heads for input and post transformers
        self.d_model_input = d_model_input
        self.nhead_input = nhead_input
        self.d_model_post = d_model_post if d_model_post is not None else d_model_input
        self.nhead_post = nhead_post if nhead_post is not None else nhead_input
        self.dim_feedforward_input = dim_feedforward_input  # Added this line
        self.dim_feedforward_post = dim_feedforward_post if dim_feedforward_post is not None else dim_feedforward_input

        # Ticker Subnetworks with input transformer configurations
        self.ticker_subnets = nn.ModuleList(
            [
                TickerSubNetwork(
                    d_model=self.d_model_input,
                    nhead=self.nhead_input,
                    num_layers=num_encoder_layers,
                    seq_len=seq_len,
                    dropout=dropout,
                    dim_feedforward=dim_feedforward_input,
                )
                for _ in range(num_tickers)
            ]
        )

        # Cross-Ticker Attention using input transformer dimensions
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.d_model_input,
            num_heads=self.nhead_input,
            dropout=dropout,
            batch_first=True,
        )

        # **Dimension Adjustment Layer (if needed)**
        if self.d_model_input != self.d_model_post:
            self.fc_dim_adjust = nn.Linear(self.d_model_input, self.d_model_post)
        else:
            self.fc_dim_adjust = nn.Identity()

        # Post-Cross-Attention Transformer Encoder with separate configurations
        self.post_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model_post,
                nhead=self.nhead_post,
                dim_feedforward=self.dim_feedforward_post,  # Updated to use self.dim_feedforward_post
                dropout=dropout,
                batch_first=True,
                layer_norm_eps=1e-4,
                activation=nn.Mish()
            ),
            num_layers=num_post_layers,
        )

        # Output Module using d_model_post dimensions
        self.fc1 = nn.Sequential(
            nn.Linear(self.d_model_post, self.d_model_post),
            nn.Mish(),
            nn.Dropout(dropout),
        )
        self.fc2 = nn.Linear(self.d_model_post, 900 * num_tickers)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            # For Transformer layers, we can rely on their default initialization

    def forward(self, minute_data):
        # Ensure inputs are lists of tensors
        if not isinstance(minute_data, list):
            minute_data = [minute_data]

        batch_size = minute_data[0].size(0)
        outputs = []

        for i in range(self.num_tickers):
            x = self.ticker_subnets[i](minute_data[i])
            outputs.append(x)

        # Stack outputs for cross-attention
        x = torch.stack(outputs, dim=1)  # Shape: (batch_size, num_tickers, d_model_input)

        # Apply cross-attention
        x, _ = self.cross_attention(x, x, x)  # x shape remains the same

        # **Adjust dimensions if necessary**
        x = self.fc_dim_adjust(x)  # Adjust to (batch_size, num_tickers, d_model_post)

        # Process through post-cross-attention transformer
        x = self.post_transformer(x)  # Processed across num_tickers dimension

        # Aggregate features
        x = torch.mean(x, dim=1)  # Shape: (batch_size, d_model_post)

        # Pass through output layers
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Update the example usage
if __name__ == "__main__":
    # Define model parameters
    num_tickers = 2
    d_model_input = 128
    nhead_input = 8
    num_encoder_layers = 2
    seq_len = 1440  # Length of 1-minute candle sequence
    dropout = 0.1

    # Initialize the model
    model = CheatCodeModel(
        num_tickers,
        d_model_input,
        nhead_input,
        num_encoder_layers,
        seq_len,
        dropout=dropout,
    )

    # Create dummy input data
    batch_size = 32
    minute_data_list = []
    for _ in range(num_tickers):
        minute_data = torch.randn(batch_size, seq_len, 15)
        minute_data_list.append(minute_data)

    # Forward pass
    output = model(minute_data_list)
    print(output.shape)  # Should be (batch_size, 1800)
