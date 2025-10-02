import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

#Cria janelas deslizantes com dados das últimas sequence_lengths de cada piloto
def create_sequences(df, feature_cols, target_col, sequence_length):
    drivers_cols = [col for col in df.columns if col.startswith('Driver_')]
    df['driver_id'] = df[drivers_cols].idxmax(axis=1)
    
    df = df.sort_values(['driver_id', 'RaceID'])
    
    previous_pieces = []
    for i in range(1, sequence_length + 1):
        shifted_features = df.groupby('driver_id')[feature_cols].shift(i)
        shifted_features.columns = [f'{col}_lag_{i}' for col in feature_cols]
        previous_pieces.append(shifted_features)
        
    df_final = pd.concat([df] + previous_pieces, axis=1)
    df_final.dropna(inplace=True)

    y = df_final[target_col].values
    
    lag_cols = [col for col in df_final.columns if 'lag' in col]
    X_flat = df_final[lag_cols].values
    
    X = X_flat.reshape(len(df_final), sequence_length, len(feature_cols))
    X = np.flip(X, axis=1)

    return X.astype(np.float32), y.astype(np.float32)

class F1Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) # Adiciona dimensão para compatibilidade

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]