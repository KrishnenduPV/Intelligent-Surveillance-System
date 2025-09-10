import torch
import torch.optim as optim
import torch.nn as nn
import pickle
import time
import json
import os
from torch.utils.data import DataLoader
from datetime import datetime

# === Define Model ===
class LSTMAnomalyDetector(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, num_layers=2, output_dim=1):
        super(LSTMAnomalyDetector, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True, 
                            bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)              # [B, T, 2*hidden_dim]
        out = self.fc(lstm_out)                 # [B, T, 1]
        out = out.mean(dim=1)                   # [B, 1] — average over time
        return out.squeeze(-1)                  # [B]


