import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

#import data 
df = pd.read_csv('PythonAPI\\data\\household_power_consumption.txt', sep=';')
print(df)

# Prepare the input and target sequences
def create_sequences(input_data, seq_length):
    # TODO Create function to prepare data 
    pass

class RNNModel(nn.Module):
    # TODO Create main RNN class for model initialization 
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNModel, self).__init__()

    def forward(self, x):
        pass

# TODO Tune hyperparameters 

input_size = None
hidden_size = None
output_size = None
num_layers = None


# Model initialization 
model = RNNModel(input_size, hidden_size, output_size, num_layers)

# TODO Selct appropriate loss criterion and optimizer 
criterion = None
optimizer = None

# TODO determine sufficient number of epochs 
num_epochs = None

# Training Loop 
for epoch in range(num_epochs):
    model.train()
    outputs = model()
    optimizer.zero_grad()
    loss = criterion(outputs, )
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')