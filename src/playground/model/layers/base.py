import torch
import torch.nn as nn
from .activation import activation_from_str


class MLP(nn.Module):
    def __init__(self, d_model, hidden_dim, activation, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.activation = activation_from_str(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Residual(nn.Module):
    """A generic residual block which can be used for input and output embedding layers."""

    def __init__(self, in_dim: int, h_dim: int, out_dim: int, dropout_p: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = nn.ReLU()
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        return out + self.residual_layer(x)
