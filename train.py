import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from preprocess import preprocess
from engine import train_epoch, evaluate
from dataset import create_sequences, F1Dataset
from model import F1_LSTM
from optimizers import CustomAdam, CustomSGD

#Realiza validação cruzada temporal, para evitar vazamento de dados.
def cross_validation(optimizer, lr):
    val_scores = []

    for fold, (train_index, val_index) in enumerate(tscv.split(X_2024)):
        print(f"\n--- Fold {fold + 1}/{config.N_SPLITS_CV} ---")
        
        X_train, X_val = X_2024[train_index], X_2024[val_index]
        y_train, y_val = y_2024[train_index], y_2024[val_index]

        train_dataset = F1Dataset(X_train, y_train)
        val_dataset = F1Dataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
        
        model = F1_LSTM().to(device)
        
        loss_fn = nn.MSELoss()
        customOptimizer = optimizer(model.parameters(), lr=lr)
        
        best_fold_loss = train_with_validation(model, customOptimizer, train_loader, val_loader, loss_fn, device, fold + 1)
        val_scores.append(best_fold_loss)
    return val_scores

#Treina o modelo no contexto da validação cruzada, verificando as perdas para cada otimizador.
def train_with_validation(model, optimizer, train_loader, val_loader, loss_fn, device, fold_num):
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        
        print(f"Fold {fold_num}, Época {epoch+1}, Perda de Treino: {train_loss:.4f}, Perda de Validação: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config.PATIENCE:
            print(f"Parada antecipada na época {epoch+1}.")
            break
            
    return best_val_loss

#Treina o modelo em todos os dados com o melhor otimizador encontrado na validação.
def train(model, optimizer, full_train_loader, loss_fn, device):
    print("\nIniciando treinamento final em todos os dados de 2024...")
    for epoch in range(config.EPOCHS):
        train_loss = train_epoch(model, full_train_loader, optimizer, loss_fn, device)
        print(f"Época Final {epoch+1}, Perda de Treino: {train_loss:.4f}")
    
    torch.save(model.state_dict(), config.SAVE_PATH)
    print(f"Modelo final treinado e salvo em: {config.SAVE_PATH}")
    return model

if __name__ == '__main__':
    df_2024 = pd.read_csv(config.PATH_2024)
    df_2025 = pd.read_csv(config.PATH_2025)
    df_2024_processed, _ = preprocess(df_2024, df_2025)
    config.INPUT_SIZE = len(df_2024_processed.columns)
    X_2024, y_2024 = create_sequences(df_2024_processed, df_2024_processed.columns, config.TARGET_COL, config.SEQUENCE_LENGTH)

    tscv = TimeSeriesSplit(n_splits=config.N_SPLITS_CV)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    val_scores_sgd = cross_validation(CustomSGD, config.LEARNING_RATE_SGD)
    print(f"\nResultado da Validação Cruzada para SGD (MSE): {np.mean(val_scores_sgd):.4f} (+/- {np.std(val_scores_sgd):.4f})")
    val_scores_adam = cross_validation(CustomAdam, config.LEARNING_RATE_ADAM)
    print(f"\nResultado da Validação Cruzada para Adam (MSE): {np.mean(val_scores_adam):.4f} (+/- {np.std(val_scores_adam):.4f})")

    full_dataset_2024 = F1Dataset(X_2024, y_2024)
    full_train_loader = DataLoader(full_dataset_2024, batch_size=config.BATCH_SIZE, shuffle=True)
    
    final_model = F1_LSTM().to(device)

    loss_fn = nn.MSELoss()
    
    #Escolhe melhor otimizador com base no MSE das validações (esperado que Adam seja melhor)
    final_optimizer = CustomAdam(final_model.parameters(), lr=config.LEARNING_RATE_ADAM) if val_scores_adam < val_scores_sgd else CustomSGD(final_model.parameters(), lr=config.LEARNING_RATE_SGD)

    final_model = train(final_model, final_optimizer, full_train_loader, loss_fn, device)

