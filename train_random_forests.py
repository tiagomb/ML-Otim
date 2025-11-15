import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
import joblib
import warnings

warnings.filterwarnings('ignore')

print("Iniciando o script de treinamento...")

# Passo 1: Leitura dos dados da Fórmula 1 de 2024
# Cria-se um DataFrame do Pandas para ajuste e leitura eficiente de colunas!
try:
    df_f2024 = pd.read_csv('formula1-2024.csv')
except FileNotFoundError:
    print("Erro: 'formula1-2024.csv' não encontrado.")
    exit()

print(f"Carregados dados de {df_f2024.shape[0]} registros de 2024.")

# Passo 2: Pré-processamento dos Dados
def clean_position(pos):
    if pos == 'NC': return 21
    try: return int(pos)
    except ValueError: return 21

df_f2024['Position'] = df_f2024['Position'].apply(clean_position)
df_f2024['Points'] = pd.to_numeric(df_f2024['Points'], errors='coerce').fillna(0)
df_f2024['Starting Grid'] = pd.to_numeric(df_f2024['Starting Grid'], errors='coerce').fillna(21) 
df_f2024['Scored_Points'] = (df_f2024['Points'] > 0).astype(int)

tracks_in_order = df_f2024['Track'].unique()
track_order_map = {track: i for i, track in enumerate(tracks_in_order)}
df_f2024['Race_Order'] = df_f2024['Track'].map(track_order_map)
df_f2024 = df_f2024.sort_values(by=['Race_Order', 'Driver'])

# Criar features de lag (baseado na Posição)
df_f2024['lag_1_pos'] = df_f2024.groupby('Driver')['Position'].shift(1)
df_f2024['lag_2_pos'] = df_f2024.groupby('Driver')['Position'].shift(2)
df_f2024['lag_3_pos'] = df_f2024.groupby('Driver')['Position'].shift(3)

# Usar todos os dados de 2024 que possuem lags como dados de treino
df_model_data = df_f2024.dropna(subset=['lag_1_pos', 'lag_2_pos', 'lag_3_pos'])

print(f"Usando {df_model_data.shape[0]} registros de 2024 (com lags) para o treino.")

# Passo 3: Features Alvos
# Aqui, define-se os atributos que devem ser previstos, como Posição e Pontos Totais
categorical_features = ['Driver', 'Team', 'Track']
numeric_features_no_grid = ['lag_1_pos', 'lag_2_pos', 'lag_3_pos']
features_no_grid = categorical_features + numeric_features_no_grid

X_train = df_model_data[features_no_grid]
y_train_pos = df_model_data['Position']
y_train_scored = df_model_data['Scored_Points']

# Encoding das variáveis categóricas
# Transforma as variáveis categórias em nuḿericas (Ex. via HotCoding)
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train_cat_ng = encoder.fit_transform(X_train[categorical_features])
X_train_final_ng = np.hstack([X_train_cat_ng, X_train[numeric_features_no_grid].values])

# Passo 4: Treinamento das Random Forests
# Random Forests de Regressão:
# Fazem a regressão da posição, baseada nas corridas anteriores.
print("Treinando Modelo 1 (Posição)...")
model_pos = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, min_samples_leaf=5)
model_pos.fit(X_train_final_ng, y_train_pos)
print("Modelo de Posição treinado.")

# Random Forests de Classificação:
# Medem a probabilidade de um piloto ter pontuado ou não, ou seja, duas classes - PONTUOU e NÃO PONTUOU
print("Treinando Modelo 2 (Pontuou?)...")
model_scored = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, min_samples_leaf=5, class_weight='balanced')
model_scored.fit(X_train_final_ng, y_train_scored)
print("Modelo 'Pontuou?' treinado.")

# Passo 06: Salvando os modelos para o teste.
print("\nSalvando modelos (treinados em 2024) para previsão...")
joblib.dump(encoder, 'f1_data_encoder.joblib')
joblib.dump(model_pos, 'f1_model_pos.joblib')
joblib.dump(model_scored, 'f1_model_scored.joblib')

print("Treinamento concluído. Modelos e encoder foram salvos.")