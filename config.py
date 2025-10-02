PATH_2024 = "formula1-2024.csv"
PATH_2025 = "formula1-2025.csv"
SAVE_PATH = "f1-model.pth"

NUMERICAL_FEATURES = [
    'Starting Grid', 'Points', 'Position', 'Total Points', 
    'Total Position', 'Team Total Points', 'Driver Contribution'
]

FEATURES_TO_DROP = ['No', 'Laps', 'Time/Retired', 'Sprint Points', 'Set Fastest Lap', 'Fastest Lap Time']

CATEGORICAL_FEATURES = ['Driver', 'Team', 'Track']
TARGET_COL = 'Points'

SEQUENCE_LENGTH = 3
BATCH_SIZE = 32
INPUT_SIZE = 0 
HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 64
DROPOUT_RATE = 0.4
LEARNING_RATE_ADAM = 0.001
LEARNING_RATE_SGD = 0.01
EPOCHS = 50
N_SPLITS_CV = 5
PATIENCE = 10 