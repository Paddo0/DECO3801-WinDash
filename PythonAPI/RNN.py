import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from torch.autograd import Variable

#training data import 
df = pd.read_csv('PythonAPI\\data\\household_power_consumption.txt', sep=';')
#print(df["Voltage"])

# Normalization, using mean 0 var 1

df['Voltage'] = pd.to_numeric(df['Voltage'], errors='coerce')
df['Voltage'] = df['Voltage'].astype(float)

scaler = MinMaxScaler(feature_range=(-1, 1))
values = df['Voltage'][:2000]
df['Voltage'] = scaler.fit_transform(df['Voltage'].values.reshape(-1,1))
V_ten = torch.FloatTensor(df['Voltage'][:2000]).view(-1)


# TODO Prepare the input and target sequences
# Define the look-back period (e.g., 60 time steps)
look_back = 120

# Create input-output pairs
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

train_sequences = create_inout_sequences(V_ten, look_back)

class LSTM(nn.Module):
    # TODO Create main RNN class for model initialization 
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.hidden_cell = (torch.zeros(2, 1, self.hidden_layer_size),
                            torch.zeros(2, 1, self.hidden_layer_size))


    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


# TODO Tune hyperparameters 

input_size = 1
hidden_size = 100
output_size = 1



# Model initialization 
model = LSTM(input_size, hidden_size, output_size)

# TODO Selct appropriate loss criterion and optimizer 
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# TODO determine sufficient number of epochs 
epochs = 20

# Training Loop 

def train(epochs):
    for i in range(epochs):
        for seq, labels in train_sequences:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = criterion(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if i % 1 == 1:
            print(f'Epoch {i} loss: {single_loss.item()}')

    print(f'Epoch {i} loss: {single_loss.item()}')

    model.eval()

    test_inputs = V_ten[-look_back:].tolist()

    for i in range(look_back):
        seq = torch.FloatTensor(test_inputs[-look_back:])
        with torch.no_grad():
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                torch.zeros(1, 1, model.hidden_layer_size))
            test_inputs.append(model(seq).item())

    # Inverse transform the predicted values to original scale
    predicted_values = scaler.inverse_transform(np.array(test_inputs[look_back:] ).reshape(-1, 1))

    # Plot the results
    import matplotlib.pyplot as plt

    plt.plot(values, label='Actual Data')
    plt.plot(np.arange(len(values), len(values) + look_back), predicted_values, label='Predictions')
    plt.legend()
    plt.show()

    
def get_pred():
    # currently a vector of zeros as test 
    return [0,0,0,0,0]

train(epochs)