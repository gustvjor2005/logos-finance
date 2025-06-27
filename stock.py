import argparse
import yfinance as yf
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class StockDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return len(self.y)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, hidden = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def create_sequence(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

def main(sequence_length):
    print(sequence_length)
    ticker = 'NVDA'
    df = yf.download(ticker, start='2024-01-01', end='2025-06-26')
    df = df[['Close']]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = create_sequence(scaled, sequence_length)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=False)
    train_dataset = StockDataset(x_train, y_train)
    test_dataset = StockDataset(x_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)

    model = LSTMModel(input_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_dataloader:
            out = model(x_batch)
            loss = criterion(out, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, loss {total_loss/len(train_dataloader):.5f}")

    # Predict last sequence
    model.eval()
    
    d = df.iloc[-sequence_length:]
    d = scaler.fit_transform(d)
    x_test = torch.tensor(d, dtype=torch.float)
    x_out = torch.unsqueeze(x_test, 0)
    
    with torch.no_grad():
        prediction = model(x_out).numpy()
        pred_original = scaler.inverse_transform(prediction)
        print(f"Prediction for last value (original scale): {pred_original[0][0]:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seq', type=int, default=30, help='Sequence length (window size)')
    args = parser.parse_args()
    main(args.seq)