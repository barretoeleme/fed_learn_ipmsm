import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from flwr.app import ArrayRecord
import datetime
from pathlib import Path

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.ReLU(),
            nn.Linear(1, 1),
            nn.ReLU(),
            nn.Linear(1, latent_dim),
            nn.ReLU() # The bottleneck layer
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.ReLU(),
            nn.Linear(1, 1),
            nn.ReLU(),
            nn.Linear(1, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded