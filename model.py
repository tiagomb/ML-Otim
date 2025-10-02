import torch.nn as nn
import config

#Inicializa modelo com os parâmetros definidos em config
class F1_LSTM(nn.Module):
    def __init__(self):
        super(F1_LSTM, self).__init__()
        self.lstm1 = nn.LSTM(config.INPUT_SIZE, config.HIDDEN_SIZE_1, batch_first=True)
        self.dropout1 = nn.Dropout(config.DROPOUT_RATE)
        self.lstm2 = nn.LSTM(config.HIDDEN_SIZE_1, config.HIDDEN_SIZE_2, batch_first=True)
        self.dropout2 = nn.Dropout(config.DROPOUT_RATE)
        self.fc1 = nn.Linear(config.HIDDEN_SIZE_2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        
        out = out[:, -1, :] 
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out