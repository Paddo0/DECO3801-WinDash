import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from torch.autograd import Variable

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load training data
df = pd.read_csv('/Users/songyutong/Downloads/household_power_consumption.txt', sep=';')

# Convert 'Global_intensity' to numeric
df['Global_intensity'] = pd.to_numeric(df['Global_intensity'], errors='coerce')
df['Global_intensity'] = df['Global_intensity'].astype(float)
prev = df['Global_intensity'][::60]

# Normalize the data
scaler = MinMaxScaler(feature_range=(-1, 1))
values = prev[:100]
df['Global_intensity'] = scaler.fit_transform(df['Global_intensity'].values.reshape(-1,1))
V_ten = torch.FloatTensor(prev[:100].to_numpy()).view(-1).to(device)

# Prepare input and target sequences
look_back = 24

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

train_sequences = create_inout_sequences(V_ten, look_back)

# LSTM Model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size).to(device)
        self.linear = nn.Linear(hidden_size, output_size).to(device)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).to(device),
                            torch.zeros(1, 1, self.hidden_layer_size).to(device))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1).to(device), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1).to(device))
        return predictions[-1]

# Hyperparameters
input_size = 1
hidden_size = 100
output_size = 1

# Initialize model
model = LSTM(input_size, hidden_size, output_size).to(device)

# Loss function and optimizer
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

# Number of epochs
epochs = 50

# Training Loop
def train(epochs):
    for i in range(epochs):
        for seq, labels in train_sequences:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                            torch.zeros(1, 1, model.hidden_layer_size).to(device))

            y_pred = model(seq)

            single_loss = criterion(y_pred, labels)
            single_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if i % 2 == 0:
            print(f'Epoch {i} loss: {single_loss.item()}')

    torch.save(model.state_dict(), 'model_weights.pth')
    print(f'Final Epoch {epochs} loss: {single_loss.item()}')

# Testing (Prediction) function
def test(inputs):
    model.load_state_dict(torch.load('model_weights.pth'))
    model.eval()

    predictions = []
    for i in range(look_back):
        seq = torch.FloatTensor(inputs[-look_back:]).to(device)
        with torch.no_grad():
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                                torch.zeros(1, 1, model.hidden_layer_size).to(device))
            inputs.append(model(seq).item())
            predictions.append(model(seq).item())

    # Inverse transform predicted values back to original scale
    predicted_values = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    print(predicted_values)

    # Plot the results
    plt.plot(scaler.inverse_transform(values.to_numpy().reshape(-1, 1)), label='Actual Data')
    plt.plot(np.arange(len(values), len(values) + look_back), predicted_values, label='Predictions')
    plt.legend()
    plt.show()
    plt.savefig('prediction.png')

# Flask Prediction API
def get_pred():
    model.load_state_dict(torch.load('model_weights.pth'))
    model.eval()

    inputs = V_ten[-look_back:].tolist()  # Use the last 'look_back' values for prediction
    for i in range(look_back):
        seq = torch.FloatTensor(inputs[-look_back:]).to(device)
        with torch.no_grad():
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                                torch.zeros(1, 1, model.hidden_layer_size).to(device))
            inputs.append(model(seq).item())
    
    predicted_values = scaler.inverse_transform(np.array(inputs[-look_back:]).reshape(-1, 1))
    return predicted_values[-1][0]  # Return the last predicted value

# Call the training process
#train(epochs)
test(V_ten[-look_back:].tolist())
