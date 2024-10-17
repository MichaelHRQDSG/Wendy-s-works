# 这是多股多维度代码
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

data = pd.read_csv('all_stocks_5yr.csv')

def normalize_stock_data(stock_name):
    stock_data = data[data['Name'] == stock_name]
    stock_data = stock_data[['date', 'open', 'high', 'low', 'close', 'volume']]
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data = stock_data.sort_values('date')
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_data[['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(stock_data[['open', 'high', 'low', 'close', 'volume']])
    return stock_data, scaler

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length][3] 
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def prepare_data(stock_names, seq_length=60):
    X, y = [], []
    scalers = {}
    for stock_name in stock_names:
        stock_data, scaler = normalize_stock_data(stock_name)
        stock_sequences, stock_labels = create_sequences(stock_data[['open', 'high', 'low', 'close', 'volume']].values, seq_length)
        X.extend(stock_sequences)
        y.extend(stock_labels)
        scalers[stock_name] = scaler
    X, y = np.array(X), np.array(y)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float().unsqueeze(-1)
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float().unsqueeze(-1)
    return X_train, y_train, X_test, y_test, scalers

def prepare_single_stock_data(stock_name, seq_length=60):
    stock_data, scaler = normalize_stock_data(stock_name)
    X, y = create_sequences(stock_data[['open', 'high', 'low', 'close', 'volume']].values, seq_length)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float().unsqueeze(-1)
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float().unsqueeze(-1)
    return X_train, y_train, X_test, y_test, scaler

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.2):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, hn = self.rnn(x, h0)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])  
        return out

def train_model(model, X_train, y_train, X_val, y_val, num_epochs, criterion, optimizer):
    model.train()
    loss_history = []
    val_loss_history = []
    
    for epoch in range(num_epochs):
        outputs = model(X_train)
        optimizer.zero_grad()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
        
        loss_history.append(loss.item())
        val_loss_history.append(val_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    
    return model, loss_history, val_loss_history

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        mse = nn.MSELoss()(predictions, y_test)
    return predictions, mse.item()

def plot_results(train_results, test_results, stock_name, train_mse, test_mse):
    plt.figure(figsize=(14, 7))
    plt.plot(train_results['True'], label='Train True')
    plt.plot(train_results['Predicted'], label='Train Predicted')
    plt.plot(range(len(train_results), len(train_results) + len(test_results)), test_results['True'], label='Test True')
    plt.plot(range(len(train_results), len(train_results) + len(test_results)), test_results['Predicted'], label='Test Predicted')
    plt.legend()
    plt.title(f'Stock Prediction for {stock_name} by RNN\nTrain MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}')
    plt.show()

stocks = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
X_train, y_train, X_test, y_test, scalers = prepare_data(stocks)

val_size = int(len(X_train) * 0.2)
X_val, y_val = X_train[-val_size:], y_train[-val_size:]
X_train, y_train = X_train[:-val_size], y_train[:-val_size]

input_size = 5  
hidden_size = 50
output_size = 1
num_epochs = 200
learning_rate = 0.001
weight_decay = 1e-5  

rnn_model = SimpleRNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(rnn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

rnn_model, train_loss_history, val_loss_history = train_model(rnn_model, X_train, y_train, X_val, y_val, num_epochs, criterion, optimizer)

predictions, mse = evaluate_model(rnn_model, X_test, y_test)
print(f'RNN Test MSE: {mse:.4f}')

def predict_and_plot_stock(stock_name):
    X_train_stock, y_train_stock, X_test_stock, y_test_stock, scaler_stock = prepare_single_stock_data(stock_name)
    
    train_predictions_stock = rnn_model(X_train_stock).detach().cpu().numpy()
    test_predictions_stock = rnn_model(X_test_stock).detach().cpu().numpy()
    
    train_mse_stock = np.mean((y_train_stock.cpu().numpy() - train_predictions_stock)**2)
    test_mse_stock = np.mean((y_test_stock.cpu().numpy() - test_predictions_stock)**2)
    
    train_results_stock = pd.DataFrame(data={'True': y_train_stock.cpu().numpy().flatten(), 'Predicted': train_predictions_stock.flatten()})
    test_results_stock = pd.DataFrame(data={'True': y_test_stock.cpu().numpy().flatten(), 'Predicted': test_predictions_stock.flatten()})
    
    plot_results(train_results_stock, test_results_stock, stock_name, train_mse_stock, test_mse_stock)

for stock in stocks:
    predict_and_plot_stock(stock)

plt.figure(figsize=(12, 6))
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()
