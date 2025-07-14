# utils/data_loader.py

import torch
from torch.utils.data import Dataset
import numpy as np

class WindPowerDataset(Dataset):
    """Custom PyTorch Dataset for wind power forecasting."""
    def __init__(self, turbine_X: np.ndarray, weather_X: np.ndarray, dynamic_features: np.ndarray, Y: np.ndarray):
        self.turbine_X = torch.from_numpy(turbine_X).float()
        self.weather_X = torch.from_numpy(weather_X).float()
        self.dynamic_features = torch.from_numpy(dynamic_features).float()
        self.Y = torch.from_numpy(Y).float()
    
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        return self.turbine_X[idx], self.weather_X[idx], self.dynamic_features[idx], self.Y[idx]