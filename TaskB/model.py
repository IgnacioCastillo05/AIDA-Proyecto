"""LSTM pequeño para pronóstico de un paso (la siguiente semana)."""
import torch
from torch import nn


class SalesLSTM(nn.Module):
    def __init__(self, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, lookback, 1) -> usamos el último estado oculto
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])
