import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

print("Iniciando o script de previsão...")

# Dicionário que define a quantidade de pontos por posição na Fórmula 1
points_map = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
    6: 8, 7: 6, 8: 4, 9: 2, 10: 1
}

# Carregamento dos modelos: Encoder e Árvores (Posição e Pontuação)
# ------------------------------------------------------------------
try:
    # Estes são os modelos SEM grid, salvos pelo script de treino
    encoder = joblib.load('f1_data_encoder.joblib')
    model_pos = joblib.load('f1_model_pos.joblib')
    model_scored = joblib.load('f1_model_scored.joblib')
except FileNotFoundError:
    print("Erro: Modelos não encontrados. Execute 'train_random_forests.py' primeiro.")
    exit()

try:
    df_2024 = pd.read_csv('formula1-2024.csv')
    df_2025 = pd.read_csv('formula1-2025.csv')
    with open('remaining-races.txt', 'r') as f:
        remaining_races_list = [line.strip() for line in f.readlines() if line.strip()]
except FileNotFoundError:
    print("Erro: Verifique se os arquivos de dados e 'remaining-races.txt' estão no diretório.")
    exit()

# Definir as colunas de Features
categorical_features = ['Driver', 'Team', 'Track']
numeric_features = ['lag_1_pos', 'lag_2_pos', 'lag_3_pos']
features = categorical_features + numeric_features

# Processamento dos Dados para efetuar os Testes:
# ---------------------------------------------------
df_history = pd.concat([df_2024, df_2025], ignore_index=True)

def clean_position(pos):
    if pos == 'NC': return 21
    try: return int(pos)
    except ValueError: return 21

df_history['Position'] = df_history['Position'].apply(clean_position)
df_history['Total Points'] = pd.to_numeric(df_history['Total Points'], errors='coerce').fillna(0)
df_history['Points'] = pd.to_numeric(df_history['Points'], errors='coerce').fillna(0)

tracks_in_order = df_history['Track'].unique()
track_order_map = {track: i for i, track in enumerate(tracks_in_order)}
df_history['Race_Order'] = df_history['Track'].map(track_order_map)
df_history = df_history.sort_values(by=['Race_Order', 'Driver'])

active_drivers_df = df_history[df_history['Race_Order'] == df_history['Race_Order'].max()]
active_drivers = active_drivers_df['Driver'].unique()
teams_map = active_drivers_df.set_index('Driver')['Team'].to_dict()

current_lags_state = {}
for driver in active_drivers:
    driver_history = df_history[df_history['Driver'] == driver].sort_values('Race_Order')
    last_3_pos = driver_history['Position'].tail(3).values.tolist()
    if len(last_3_pos) < 3:
        last_3_pos = [21] * (3 - len(last_3_pos)) + last_3_pos
    current_lags_state[driver] = last_3_pos

current_total_points_map = active_drivers_df.set_index('Driver')['Total Points'].to_dict()
for driver in active_drivers:
    if driver not in current_total_points_map:
        current_total_points_map[driver] = 0

print(f"Estado inicial dos pilotos carregado. Iniciando previsão para {len(remaining_races_list)} corridas...")
all_predictions_list = []

# Loop de Previsão de Corridas
# Utiliza as últimas corridas e as informações de posições anteriores
# para estimar a próxima corrida. A previsão é usada como dado para a
# próxima corrida a ser prevista, até que não sobre mais corridas para
# prever.
# ---------------------------------------------------------------------
for track in remaining_races_list:
    X_pred_race_data = []
    drivers_in_this_race = []
    
    for driver in active_drivers:
        lags = current_lags_state[driver]
        team = teams_map.get(driver, 'Unknown')
        X_pred_race_data.append([driver, team, track, lags[2], lags[1], lags[0]])
        drivers_in_this_race.append(driver)

    X_pred_df = pd.DataFrame(X_pred_race_data, columns=features)
    
    X_pred_cat = encoder.transform(X_pred_df[categorical_features])
    X_pred_num = X_pred_df[numeric_features].values
    X_pred_final = np.hstack([X_pred_cat, X_pred_num])
    
    predicted_positions_float = model_pos.predict(X_pred_final)
    predicted_scored_binary = model_scored.predict(X_pred_final)
    
    race_results = pd.DataFrame({
        'Track': track,
        'Driver': drivers_in_this_race,
        'Predicted_Position_Float': predicted_positions_float,
        'Will_Score_Binary': predicted_scored_binary
    })
    

    race_results['Race_Predicted_Position'] = race_results['Predicted_Position_Float'].rank().astype(int)
    
    race_results['Race_Predicted_Points'] = race_results['Race_Predicted_Position'].map(points_map).fillna(0).astype(int)
    
    race_results['Race_Will_Score'] = race_results['Will_Score_Binary'].map({0: 'Não', 1: 'Sim'})

    race_results['Championship_Points_Before_Race'] = race_results['Driver'].map(current_total_points_map)
    race_results['Championship_Total_Points'] = race_results['Championship_Points_Before_Race'] + race_results['Race_Predicted_Points']
    race_results['Championship_Total_Position'] = race_results['Championship_Total_Points'].rank(method='dense', ascending=False).astype(int)
    
    new_positions_map = race_results.set_index('Driver')['Race_Predicted_Position'].to_dict()
    current_total_points_map = race_results.set_index('Driver')['Championship_Total_Points'].to_dict()
    
    for driver in active_drivers:
        current_lags_state[driver].pop(0)
        current_lags_state[driver].append(new_positions_map[driver])
    
    all_predictions_list.append(race_results[['Track', 'Driver', 
                                              'Race_Predicted_Position', 'Race_Predicted_Points', 'Race_Will_Score',
                                              'Championship_Total_Points', 'Championship_Total_Position']])

# Salvar os Resultados finais em um .csv
# --------------------------------------------------------
final_predictions_df = pd.concat(all_predictions_list)
final_predictions_df['Track'] = pd.Categorical(final_predictions_df['Track'], categories=remaining_races_list, ordered=True)
final_predictions_df = final_predictions_df.sort_values(by=['Track', 'Championship_Total_Position', 'Race_Predicted_Position'])

csv_filename = 'predicted_season_standings.csv'

int_columns = ['Race_Predicted_Position', 'Race_Predicted_Points', 'Championship_Total_Points', 'Championship_Total_Position']
final_predictions_df[int_columns] = final_predictions_df[int_columns].astype(int)
final_predictions_df.to_csv(csv_filename, index=False)

print(f"\nPrevisões concluídas e salvas em '{csv_filename}'.")