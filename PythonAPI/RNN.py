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

# LSTM Model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=self.num_layers).to(device)
        self.linear = nn.Linear(hidden_size, output_size).to(device)
        self.hidden_cell = (torch.zeros(self.num_layers, 1, self.hidden_layer_size).to(device),
                            torch.zeros(self.num_layers, 1, self.hidden_layer_size).to(device))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1).to(device), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1).to(device))
        return predictions[-1]

# Hyperparameters
input_size = 1
hidden_size = 75
output_size = 1
num_layers = 1  # Number of LSTM layers

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

def test(inputs):
    global predicted_values_global  # Global variable to store prediction values
    model.load_state_dict(torch.load('model_weights.pth'))  # Load the trained model
    model.eval()

    predictions = []
    for i in range(look_back):
        seq = torch.FloatTensor(inputs[-look_back:]).to(device)  # Use the input data
        with torch.no_grad():
            model.hidden_cell = (torch.zeros(num_layers, 1, model.hidden_layer_size).to(device),
                                torch.zeros(num_layers, 1, model.hidden_layer_size).to(device))
            pred_value = model(seq).item()
            inputs.append(pred_value)
            predictions.append(pred_value)
    
    # Save the prediction result
    predicted_values_global = np.array(predictions).reshape(-1, 1)  # Store in global result
    print("Predicted Values:\n", predicted_values_global)
    
    # Plot the actual data and prediction
    
    # There are many risks when plotting(it may block your backend progressing) you can uncomment this part when you train to show the plot
    """
    actual_data = values.to_numpy().reshape(-1, 1)  # Use the original data
    plt.plot(actual_data, label='Actual Data', color='blue')  # Actual data is shown in blue
    plt.plot(np.arange(len(values), len(values) + look_back), predicted_values_global, label='Predictions', color='red')  # Prediction data is shown in red
    plt.axvline(x=len(values), color='red', linestyle='--', label='Prediction Start')  # Mark where predictions start
    plt.legend()

    # Save the plot as a file instead of displaying it
    plt.savefig('prediction.png')
    plt.close()  # Ensure Matplotlib image is closed to prevent it from trying to manipulate a graphical window in the background thread
    """

    return predictions

# Call the training process
# train(epochs)

# Call the test process
# test(V_ten[-look_back:].tolist())
