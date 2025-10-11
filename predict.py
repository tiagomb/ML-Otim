import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import config
from preprocess import preprocess
from model import F1_LSTM
from dataset import create_sequences

#Cria todas as colunas e linhas de uma nova corrida, simulando os dados com base nos pontos previstos.
#Além disso, concatena-as aos datasets não normalizados e normalizados, para serem utilizadas nas previsões seguintes
def create_and_append_race_results(predictions_df, current_season_unscaled_df, current_season_df, last_known_race_id, track_for_prediction, scaler, df_2025_processed):
    
    last_race_state = current_season_unscaled_df.groupby('Driver ID').last().reset_index()
    columns_to_carry = ['Driver ID', 'Driver', 'Team', 'Total Points', 'Total Position', 'Team Total Points', 'Season']
    new_race_df = pd.merge(last_race_state[columns_to_carry],
                           predictions_df[['Driver ID', 'Starting Grid', 'Position', 'Points']],
                           on='Driver ID')

    new_race_df['RaceID'] = last_known_race_id + 1
    new_race_df['Track'] = track_for_prediction
    new_race_df['Total Points'] += new_race_df['Points']
    new_race_df = new_race_df.sort_values('Total Points', ascending=False)
    new_race_df['Total Position'] = range(1, len(new_race_df) + 1)

    team_race_points = new_race_df.groupby('Team')['Points'].sum().reset_index()
    last_team_points = current_season_unscaled_df.groupby('Team')['Team Total Points'].last().reset_index()
    merged_teams = pd.merge(last_team_points, team_race_points, on='Team', how='left').fillna(0)
    merged_teams['Team Total Points'] += merged_teams['Points']
    new_race_df = new_race_df.drop(columns=['Team Total Points'])
    new_race_df = pd.merge(new_race_df, merged_teams[['Team', 'Team Total Points']], on='Team', how='left')
    new_race_df['Driver Contribution'] = np.where(new_race_df['Team Total Points'] > 0, new_race_df['Points'] / new_race_df['Team Total Points'], 0)

    updated_unscaled_df = pd.concat([current_season_unscaled_df, new_race_df[current_season_unscaled_df.columns]], ignore_index=True)

    new_race_processed_df = new_race_df.copy()
    new_race_processed_df = pd.get_dummies(new_race_processed_df, columns=config.CATEGORICAL_FEATURES, drop_first=True)
    new_race_processed_df = new_race_processed_df.reindex(columns=df_2025_processed.columns, fill_value=0)
    numerical_features_in_df = [col for col in config.NUMERICAL_FEATURES if col in new_race_processed_df.columns]
    new_race_processed_df[numerical_features_in_df] = scaler.transform(new_race_processed_df[numerical_features_in_df])
    new_race_processed_df['Driver ID'] = new_race_df['Driver ID']
    
    updated_processed_df = pd.concat([current_season_df, new_race_processed_df], ignore_index=True)

    return updated_unscaled_df, updated_processed_df

#Prevê as N corridas seguintes, usando como alvo a pontuação
def predict_future_races(n_races_to_predict: int):
    df_2024 = pd.read_csv(config.PATH_2024)
    df_2025 = pd.read_csv(config.PATH_2025)

    _, df_2025_processed, df_2025_unscaled, scaler = preprocess(df_2024, df_2025)

    config.INPUT_SIZE = len(df_2025_processed.columns)

    driver_cols_one_hot = [col for col in df_2025_processed.columns if col.startswith('Driver_')]
    df_2025_processed['Driver ID'] = df_2025_processed[driver_cols_one_hot].idxmax(axis=1)
    df_2025_unscaled['Driver ID'] = df_2025_processed['Driver ID']

    current_season_df = df_2025_processed.copy()
    current_season_unscaled_df = df_2025_unscaled.copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = F1_LSTM().to(device)
    model.load_state_dict(torch.load(config.SAVE_PATH, map_location=device))
    model.eval()

    races_2025_calendar = [
        "Australia", "China", "Japan", "Bahrain", "Saudi Arabia", "Miami", "Emilia-Romagna", "Monaco", 
        "Spain", "Canada", "Austria", "Great Britain", "Belgium", "Hungary", 
        "Netherlands", "Italy", "Azerbaijan", "Singapore", "United States", "Mexico", 
        "Brazil", "Las Vegas", "Qatar", "Abu Dhabi"
    ]

    available_tracks = [track for track in races_2025_calendar if track not in current_season_unscaled_df['Track'].unique()]

    if n_races_to_predict > len(available_tracks):
        raise ValueError(f"Não é possível prever {n_races_to_predict} corridas. Apenas {len(available_tracks)} estão disponíveis.")

    points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    last_known_race_id = current_season_df['RaceID'].max() - 23 #Desconsidera corridas de 2024

    for i in range(n_races_to_predict):
        track_for_prediction = available_tracks[i]
        last_known_race_id+=1
        print(f"-> Prevendo Corrida {last_known_race_id} (Pista: {track_for_prediction})...")

        feature_columns_for_sequence = [col for col in current_season_df.columns if col != 'Driver ID']
        X_pred, _ = create_sequences(current_season_df.copy(), feature_columns_for_sequence, config.TARGET_COL, config.SEQUENCE_LENGTH)

        df_for_order = current_season_df.copy()
        df_for_order['driver_id'] = df_for_order[driver_cols_one_hot].idxmax(axis=1)
        df_for_order = df_for_order.sort_values(['driver_id', 'RaceID'])
        
        #Mesma lógica do create_sequences
        previous_pieces = []
        for j in range(1, config.SEQUENCE_LENGTH + 1):
            shifted_features = df_for_order.groupby('driver_id')[feature_columns_for_sequence].shift(j)
            previous_pieces.append(shifted_features)
        
        df_final_recreated = pd.concat([df_for_order] + previous_pieces, axis=1)
        df_final_recreated.dropna(inplace=True)
        driver_ids_per_sequence = df_final_recreated['driver_id']

        with torch.no_grad():
            X_pred_tensor = torch.from_numpy(X_pred).float().to(device)
            output = model(X_pred_tensor).cpu().numpy().flatten()

        noise = np.random.normal(0, 0.25, len(output))
        
        sequence_predictions = pd.DataFrame({'Driver ID': driver_ids_per_sequence.values, 'Predicted Points': output+noise})
        final_predictions = sequence_predictions.groupby('Driver ID').last()

        predictions_df = final_predictions.sort_values('Predicted Points', ascending=False).reset_index()
        predictions_df['Position'] = range(1, len(predictions_df) + 1)
        predictions_df['Points'] = predictions_df['Position'].map(points_map).fillna(0)
        predictions_df['Starting Grid'] = np.random.permutation(range(1, len(predictions_df) + 1))

        current_season_unscaled_df, current_season_df = create_and_append_race_results(
            predictions_df,
            current_season_unscaled_df,
            current_season_df,
            last_known_race_id,
            track_for_prediction,
            scaler,
            df_2025_processed
        )

    final_standings = current_season_unscaled_df.groupby('Driver ID').last().sort_values('Total Points', ascending=False).reset_index()
    final_standings['Final Position'] = range(1, len(final_standings) + 1)

    print(f"\n--- Previsão do Campeonato Após {last_known_race_id + 1} Corridas ---")
    print(final_standings[['Final Position', 'Driver', 'Team', 'Total Points']])

if __name__ == '__main__':
    N_RACES = 9
    predict_future_races(N_RACES)