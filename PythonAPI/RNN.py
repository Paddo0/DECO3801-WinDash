import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load training data
df = pd.read_csv('data.csv', sep=';') # replace with your local training dataset

# Convert 'Global_intensity' to float
df['Global_intensity'] = pd.to_numeric(df['Global_intensity'], errors='coerce')
df['Global_intensity'] = df['Global_intensity'].astype(float)
df['Global_intensity'] = [np.mean(df['Global_intensity'][i*60:i*60 + 60]) for i in range(len(df['Global_intensity']/60 - 1))]


# Downsample the data, taking one value per minute
prev = df['Global_intensity']

# Get the first 100 values for training
values = prev[:100]
# Convert the first 100 values to a tensor and move it to GPU (if available)
V_ten = torch.FloatTensor(prev[:100].to_numpy()).view(-1).to(device)

# Prepare input and target sequences

look_back = 24 # For next 24 hours use

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

train_sequences = create_inout_sequences(V_ten, look_back)



# Initialize the model
model = LSTM(input_size, hidden_size, output_size, num_layers=num_layers).to(device)

# Loss function and optimizer
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

# Training loop
epochs = 50

def train(epochs):
    for epoch in range(epochs):
        total_loss = 0
        for seq, labels in train_sequences:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(num_layers, 1, model.hidden_layer_size).to(device),
                                 torch.zeros(num_layers, 1, model.hidden_layer_size).to(device))

            y_pred = model(seq)

            single_loss = criterion(y_pred, labels)
            single_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += single_loss.item()
        
        average_loss = total_loss / len(train_sequences)
        print(f'Epoch {epoch}, Average Loss: {average_loss}')

    torch.save(model.state_dict(), 'model_weights.pth')
    print(f'Final Epoch {epochs} loss: {average_loss}')

# Test function
predicted_values_global = None  # Global variable to store prediction values


# Call the training process
# train(epochs)

# Call the test process
# test(V_ten[-look_back:].tolist())
