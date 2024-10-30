import torch
import torch.nn as nn
from preprocess import preprocess_data
from torch.utils.data import DataLoader, TensorDataset

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[-1])
        return out

def train_lstm_model(X_train, y_train):
    model = LSTMModel(input_size=100, hidden_size=50, output_size=1)
    # Training logic for LSTM here
    # DataLoader, loss function, optimizer, etc.
    return model

def predict_lstm(model, X_test):
    # Inference logic for LSTM model
    predictions = model(X_test)
    return predictions

def run_lstm(df, text_column, target_column):
    X_train, X_test, y_train, y_test = preprocess_data(df, text_column, target_column)
    model = train_lstm_model(X_train, y_train)
    predictions = predict_lstm(model, X_test)
    return predictions, y_test
