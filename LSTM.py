import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('all_stocks_5yr.csv')

# Create sequences function
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length][3]  # 'close' is the target variable
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Prepare data for multiple stocks
def prepare_data(stock_names, seq_length=60):
    sequences = []
    scalers = {}
    for stock in stock_names:
        stock_data = data[data['Name'] == stock]
        stock_data = stock_data[['date', 'open', 'high', 'low', 'close', 'volume']]
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        stock_data = stock_data.sort_values('date')
        scaler = MinMaxScaler(feature_range=(0, 1))
        stock_data[['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(stock_data[['open', 'high', 'low', 'close', 'volume']])
        stock_sequences, stock_labels = create_sequences(stock_data[['open', 'high', 'low', 'close', 'volume']].values, seq_length)
        sequences.extend(list(zip(stock_sequences, stock_labels)))
        scalers[stock] = scaler
    
    X, y = zip(*sequences)
    X, y = np.array(X), np.array(y)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float().unsqueeze(-1)
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float().unsqueeze(-1)
    return X_train, y_train, X_test, y_test, scalers

# Prepare data for a single stock
def prepare_single_stock_data(stock_name, seq_length=60):
    stock_data = data[data['Name'] == stock_name]
    stock_data = stock_data[['date', 'open', 'high', 'low', 'close', 'volume']]
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data = stock_data.sort_values('date')
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_data[['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(stock_data[['open', 'high', 'low', 'close', 'volume']])
    X, y = create_sequences(stock_data[['open', 'high', 'low', 'close', 'volume']].values, seq_length)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float().unsqueeze(-1)
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float().unsqueeze(-1)
    return X_train, y_train, X_test, y_test, scaler

# Define the LSTM model
class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.2):
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.Wxf = nn.Linear(input_size, hidden_size)
        self.Whf = nn.Linear(hidden_size, hidden_size)
        self.bf = nn.Parameter(torch.zeros(hidden_size))
        self.Wxi = nn.Linear(input_size, hidden_size)
        self.Whi = nn.Linear(hidden_size, hidden_size)
        self.bi = nn.Parameter(torch.zeros(hidden_size))
        self.Wxc = nn.Linear(input_size, hidden_size)
        self.Whc = nn.Linear(hidden_size, hidden_size)
        self.bc = nn.Parameter(torch.zeros(hidden_size))
        self.Wxo = nn.Linear(input_size, hidden_size)
        self.Who = nn.Linear(hidden_size, hidden_size)
        self.bo = nn.Parameter(torch.zeros(hidden_size))
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        for t in range(seq_len):
            x_t = x[:, t, :]
            f_t = torch.sigmoid(self.Wxf(x_t) + self.Whf(h_t) + self.bf)
            i_t = torch.sigmoid(self.Wxi(x_t) + self.Whi(h_t) + self.bi)
            c_tilde_t = torch.tanh(self.Wxc(x_t) + self.Whc(h_t) + self.bc)
            c_t = f_t * c_t + i_t * c_tilde_t
            o_t = torch.sigmoid(self.Wxo(x_t) + self.Who(h_t) + self.bo)
            h_t = o_t * torch.tanh(c_t)
            h_t = self.dropout(h_t)
        out = self.fc(h_t)
        return out

# Train the model
def train_model(model, X_train, y_train, num_epochs=400, learning_rate=0.001, weight_decay=1e-5):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_history = []
    
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        optimizer.zero_grad()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return model, loss_history

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        mse = nn.MSELoss()(predictions, y_test)
    return predictions, mse.item()

# Inverse transform predictions and true values
def inverse_transform(scaler, data):
    data = scaler.inverse_transform(data.detach().cpu().numpy())
    return data

# Plot the results
def plot_results(train_results, test_results, stock_name,train_mse,test_mse):
    plt.figure(figsize=(14, 7))
    plt.plot(train_results['True'], label='Train True')
    plt.plot(train_results['Predicted'], label='Train Predicted')
    plt.plot(range(len(train_results), len(train_results) + len(test_results)), test_results['True'], label='Test True')
    plt.plot(range(len(train_results), len(train_results) + len(test_results)), test_results['Predicted'], label='Test Predicted')
    plt.legend()
    plt.title(f'Stock Prediction for {stock_name} by RNN\nTrain MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}')
    plt.show()

# Evaluate and plot function
def evaluate_and_plot(stock_name, model, seq_length=60):
    X_train_stock, y_train_stock, X_test_stock, y_test_stock, scaler_stock = prepare_single_stock_data(stock_name, seq_length)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_stock, y_train_stock = X_train_stock.to(device), y_train_stock.to(device)
    X_test_stock, y_test_stock = X_test_stock.to(device), y_test_stock.to(device)

    model.eval()
    with torch.no_grad():
        train_predictions_stock = model(X_train_stock)
        test_predictions_stock = model(X_test_stock)
    
    train_results_stock = pd.DataFrame(data={'True': y_train_stock.cpu().numpy().flatten(), 'Predicted': train_predictions_stock.cpu().numpy().flatten()})
    test_results_stock = pd.DataFrame(data={'True': y_test_stock.cpu().numpy().flatten(), 'Predicted': test_predictions_stock.cpu().numpy().flatten()})

    train_mse_stock = np.mean((train_results_stock['True'] - train_results_stock['Predicted'])**2)
    test_mse_stock = np.mean((test_results_stock['True'] - test_results_stock['Predicted'])**2)

    plot_results(train_results_stock, test_results_stock, stock_name,train_mse_stock,test_mse_stock)


# Prepare data for multiple stocks for training
stocks = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
X_train, y_train, X_test, y_test, scalers = prepare_data(stocks)

# Initialize and train the LSTM model
input_size = 5  # Updated for 5 features: 'open', 'high', 'low', 'close', 'volume'
hidden_size = 50
output_size = 1

lstm_model = CustomLSTM(input_size, hidden_size, output_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_model.to(device)
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

lstm_model, train_loss_history = train_model(lstm_model, X_train, y_train)

# Evaluate the model on test set
predictions, mse = evaluate_model(lstm_model, X_test, y_test)
print(f'Test MSE: {mse:.4f}')

# Use the evaluate_and_plot function to evaluate and plot results for AAPL and other individual stocks
evaluate_and_plot('AAPL', lstm_model)
evaluate_and_plot('MSFT', lstm_model)
evaluate_and_plot('GOOG', lstm_model)
evaluate_and_plot('AMZN', lstm_model)

# Plot training loss
plt.figure(figsize=(12, 6))
plt.plot(train_loss_history, label='Train Loss')
plt.legend()
plt.title('Training Loss')
plt.show()