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

def set_seed(seed=5):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Garante determinismo em operações de convolução/RNN no CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed fixada em: {seed}")

def cross_validation(optimizer, lr, **optimizer_kwargs):
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
        customOptimizer = optimizer(model.parameters(), lr=lr, **optimizer_kwargs)
        
        best_fold_loss = train_with_validation(model, customOptimizer, train_loader, val_loader, loss_fn, device)
        val_scores.append(best_fold_loss)
    return val_scores

def train_with_validation(model, optimizer, train_loader, val_loader, loss_fn, device):
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.EPOCHS):
        train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config.PATIENCE:
            print(f"Parada antecipada na época {epoch+1}.")
            break
            
    return best_val_loss

def train(model, optimizer, full_train_loader, loss_fn, device):
    print("\nIniciando treinamento final em todos os dados de 2024...")
    for _ in range(config.EPOCHS):
        train_epoch(model, full_train_loader, optimizer, loss_fn, device)
    
    torch.save(model.state_dict(), config.SAVE_PATH)
    print(f"Modelo final treinado e salvo em: {config.SAVE_PATH}")
    return model

if __name__ == '__main__':
    set_seed()
    df_2024 = pd.read_csv(config.PATH_2024)
    df_2025 = pd.read_csv(config.PATH_2025)
    df_2024_processed, _, _, _ = preprocess(df_2024, df_2025)
    config.INPUT_SIZE = len(df_2024_processed.columns)
    X_2024, y_2024 = create_sequences(df_2024_processed, df_2024_processed.columns, config.TARGET_COL, config.SEQUENCE_LENGTH)

    tscv = TimeSeriesSplit(n_splits=config.N_SPLITS_CV)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    cv_results = {}

    # 1. SGD
    print("\n--- Iniciando Validação Cruzada: SGD ---")
    val_scores_sgd = cross_validation(CustomSGD, config.LEARNING_RATE_SGD, momentum=0.0)
    mean_sgd = np.mean(val_scores_sgd)
    cv_results['sgd'] = {'mean_loss': mean_sgd, 'optimizer': CustomSGD, 'lr': config.LEARNING_RATE_SGD, 'kwargs': {'momentum': 0.0}}
    print(f"\nResultado da Validação Cruzada para SGD (MSE): {mean_sgd:.4f} (+/- {np.std(val_scores_sgd):.4f})")

    # 2. SGD com Momentum
    print("\n--- Iniciando Validação Cruzada: SGD com Momentum ---")
    val_scores_sgd_momentum = cross_validation(CustomSGD, config.LEARNING_RATE_SGD, momentum=config.MOMENTUM)
    mean_sgd_momentum = np.mean(val_scores_sgd_momentum)
    cv_results['sgd_momentum'] = {'mean_loss': mean_sgd_momentum, 'optimizer': CustomSGD, 'lr': config.LEARNING_RATE_SGD, 'kwargs': {'momentum': config.MOMENTUM}}
    print(f"\nResultado da Validação Cruzada para SGD com Momentum (MSE): {mean_sgd_momentum:.4f} (+/- {np.std(val_scores_sgd_momentum):.4f})")

    # 3. Adam
    print("\n--- Iniciando Validação Cruzada: Adam ---")
    val_scores_adam = cross_validation(CustomAdam, config.LEARNING_RATE_ADAM, weight_decay=0.0)
    mean_adam = np.mean(val_scores_adam)
    cv_results['adam'] = {'mean_loss': mean_adam, 'optimizer': CustomAdam, 'lr': config.LEARNING_RATE_ADAM, 'kwargs': {'weight_decay': 0.0}}
    print(f"\nResultado da Validação Cruzada para Adam (MSE): {mean_adam:.4f} (+/- {np.std(val_scores_adam):.4f})")

    # 4. AdamW
    print("\n--- Iniciando Validação Cruzada: AdamW ---")
    val_scores_adamw = cross_validation(CustomAdam, config.LEARNING_RATE_ADAM, weight_decay=config.WEIGHT_DECAY)
    mean_adamw = np.mean(val_scores_adamw)
    cv_results['adamw'] = {'mean_loss': mean_adamw, 'optimizer': CustomAdam, 'lr': config.LEARNING_RATE_ADAM, 'kwargs': {'weight_decay': config.WEIGHT_DECAY}}
    print(f"\nResultado da Validação Cruzada para AdamW (MSE): {mean_adamw:.4f} (+/- {np.std(val_scores_adamw):.4f})")


    full_dataset_2024 = F1Dataset(X_2024, y_2024)
    full_train_loader = DataLoader(full_dataset_2024, batch_size=config.BATCH_SIZE, shuffle=True)
    
    final_model = F1_LSTM().to(device)
    loss_fn = nn.MSELoss()
    
    # Escolhe o melhor otimizador com base no menor MSE da validação cruzada
    best_optimizer_name = min(cv_results, key=lambda k: cv_results[k]['mean_loss'])
    best_config = cv_results[best_optimizer_name]
    
    print(f"\n--- Melhor otimizador escolhido: {best_optimizer_name.upper()} ---")
    print(f"MSE médio na validação cruzada: {best_config['mean_loss']:.4f}")
    
    best_optimizer_class = best_config['optimizer']
    best_lr = best_config['lr']
    best_kwargs = best_config['kwargs']
    
    # Instancia o otimizador final com os melhores parâmetros encontrados
    final_optimizer = best_optimizer_class(final_model.parameters(), lr=best_lr, **best_kwargs)
    
    # Treina o modelo final
    final_model = train(final_model, final_optimizer, full_train_loader, loss_fn, device)