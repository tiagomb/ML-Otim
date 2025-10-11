import pandas as pd
import numpy as np
import config
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#Realiza pré-processamento dos dados. É necessário unir ambos os datasets, para garantir que os modelos de treinamento/predição possuam tamanhos compatíveis
#Aqui, colunas desnecessárias são removidas, variáveis são normalizadas/codificadas e features novas são criadas, conforme necessidade
def preprocess(df_train, df_predict):
    df_train['Season'] = 2024
    df_predict['Season'] = 2025
    df_train = df_train.drop(config.FEATURES_TO_DROP, axis=1)
    df_predict = df_predict.drop(config.FEATURES_TO_DROP, axis=1)
    df_train['RaceID'] = pd.factorize(df_train['Track'])[0]

    full_df = pd.concat([df_train, df_predict], ignore_index=True)
    full_df = full_df[full_df['Driver'] != 'Jack Doohan'].reset_index(drop=True)
    full_df['RaceID'] = pd.factorize(full_df['Season'].astype(str) + '-' + full_df['Track'])[0]
    full_df.sort_values('RaceID', inplace=True)

    team_race_points = full_df.groupby(['RaceID', 'Team'])['Points'].sum().reset_index()
    team_race_points.rename(columns={'Points': 'Team Points'}, inplace=True)
    team_race_points = team_race_points.sort_values('RaceID')
    team_race_points['Team Total Points'] = team_race_points.groupby('Team')['Team Points'].cumsum()

    full_df = pd.merge(full_df, team_race_points[['RaceID', 'Team', 'Team Total Points']], on=['RaceID', 'Team'], how='left')
    full_df['Driver Contribution'] = np.where(full_df['Team Total Points'] > 0, full_df['Total Points'] / full_df['Team Total Points'], 0)
    full_df['Position'] = pd.to_numeric(full_df['Position'], errors='coerce').fillna(21)

    df_predict_unscaled = full_df[full_df['Season'] == 2025].copy()

    full_df = pd.get_dummies(full_df, columns=config.CATEGORICAL_FEATURES, drop_first=False)

    scaler = StandardScaler()
    full_df[config.NUMERICAL_FEATURES] = scaler.fit_transform(full_df[config.NUMERICAL_FEATURES])

    df_train_processed = full_df[full_df['Season'] == 2024]
    df_predict_processed = full_df[full_df['Season'] == 2025]

    df_train_processed = df_train_processed.drop(['Season'], axis=1)
    df_predict_processed = df_predict_processed.drop(['Season'], axis=1)
    
    return df_train_processed, df_predict_processed, df_predict_unscaled, scaler