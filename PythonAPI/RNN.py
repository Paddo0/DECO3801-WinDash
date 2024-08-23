import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


#training data import 
df = pd.read_csv('PythonAPI\\data\\household_power_consumption.txt', sep=';')
#print(df["Voltage"])

# Normalization, using mean 0 var 1

df['Voltage'] = pd.to_numeric(df['Voltage'], errors='coerce')
df['Voltage'] = df['Voltage'].astype(float)

scaler = MinMaxScaler(feature_range=(-1, 1))
df['Voltage'] = scaler.fit_transform(df['Voltage'].values.reshape(-1,1))

# TODO Prepare the input and target sequences
def create_sequences(data, slen):
    sequences = []
    targets = []
    for i in range(len(data) - slen):
        seq = data[i:i + slen]
        target = data[i + slen]
        sequences.append(seq)
        targets.append(target)
    return torch.FloatTensor(sequences), torch.FloatTensor(targets)

class RNNModel(nn.Module):
    # TODO Create main RNN class for model initialization 
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        
        self.fc = nn.Linear(hidden_size, output_size)


def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

    # Forward propagation
    out, _ = self.lstm(x, (h0.detach(), c0.detach()))

    # Vary output depending on how many predictions we need 
    out = self.fc(out[:, -1, :])
    return out


# TODO Tune hyperparameters 

input_size = 0
hidden_size = 0
output_size = 0
num_layers = 0


# Model initialization 
#model = RNNModel(input_size, hidden_size, output_size, num_layers)

# TODO Selct appropriate loss criterion and optimizer 
criterion = None
optimizer = None

# TODO determine sufficient number of epochs 
num_epochs = None

# Training Loop 
def train(model, criterion, optimizer):
    for epoch in range(num_epochs):
        model.train()
        outputs = model()
        optimizer.zero_grad()
        loss = criterion(outputs, )
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.2f}')

def get_pred():
    # currently a vector of zeros as test 
    return [0,0,0,0,0]