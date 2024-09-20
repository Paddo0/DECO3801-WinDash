import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error, mean_absolute_error  # Import for error calculation

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load training data
df = pd.read_csv('/Users/songyutong/Downloads/household_power_consumption.txt', sep=';')

# Convert 'Global_active_power' column to numeric, set non-convertible values to NaN
df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
# Convert 'Global_active_power' column to float
df['Global_active_power'] = df['Global_active_power'].astype(float)
# Downsample the data, taking one value per minute
prev = df['Global_active_power'][::60]

# Normalize the data
scaler = MinMaxScaler(feature_range=(-1, 1))
# Use the first 100 values for training
values = prev[:100]
# Apply scaling on 'Global_active_power'
df['Global_active_power'] = scaler.fit_transform(df['Global_active_power'].values.reshape(-1,1))
# Convert the first 100 values to a tensor and move to GPU (if available)
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

# LSTM Model with increased number of layers
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
hidden_size = 100
output_size = 1
num_layers = 3  # Increase the number of LSTM layers

# Initialize model with more layers
model = LSTM(input_size, hidden_size, output_size, num_layers=num_layers).to(device)

# Loss function and optimizer
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

# Increase the number of epochs
epochs = 100  # Increase the training epochs

# Training Loop with loss printing
def train(epochs):
    for epoch in range(epochs):
        total_loss = 0  # Initialize total loss for the epoch
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

            # Accumulate total loss for the epoch
            total_loss += single_loss.item()
        
        # Calculate and print the average loss for this epoch
        average_loss = total_loss / len(train_sequences)
        print(f'Epoch {epoch}, Average Loss: {average_loss}')

    torch.save(model.state_dict(), 'model_weights.pth')
    print(f'Final Epoch {epochs} loss: {average_loss}')

# Testing (Prediction) function with error printing
# Global variable to store predictions generated in the test function
predicted_values_global = None

# Test function to generate predictions and save them to the global variable
def test(inputs):
    global predicted_values_global  # Use global variable
    model.load_state_dict(torch.load('model_weights.pth'))
    model.eval()

    predictions = []
    for i in range(look_back):
        seq = torch.FloatTensor(inputs[-look_back:]).to(device)
        with torch.no_grad():
            model.hidden_cell = (torch.zeros(num_layers, 1, model.hidden_layer_size).to(device),
                                torch.zeros(num_layers, 1, model.hidden_layer_size).to(device))
            pred_value = model(seq).item()
            inputs.append(pred_value)
            predictions.append(pred_value)

    # Inverse transform predicted values back to original scale
    predicted_values = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    predicted_values_global = predicted_values  # Save predictions to the global variable
    print("Predicted Values:\n", predicted_values)

    # Combine actual data and predicted data
    actual_data = scaler.inverse_transform(values.to_numpy().reshape(-1, 1))
    combined_values = np.concatenate((actual_data, predicted_values))

    # Calculate and print prediction errors
    mse = mean_squared_error(actual_data[-look_back:], predicted_values)
    mae = mean_absolute_error(actual_data[-look_back:], predicted_values)
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")

    # Plot the results
    plt.plot(actual_data, label='Actual Data', color='blue')  # Set actual data color to blue
    plt.plot(np.arange(len(values), len(values) + look_back), predicted_values, label='Predictions', color='red')  # Set predicted data color to red
    plt.axvline(x=len(values), color='red', linestyle='--', label='Prediction Start')  # Optional: Mark the start of predictions
    plt.legend()
    plt.show()
    plt.savefig('prediction.png')

# Flask Prediction API directly returns the first prediction generated by the test function
def get_pred():
    global predicted_values_global
    if predicted_values_global is not None:
        return predicted_values_global[0][0]  # Return the first prediction value
    else:
        return "No prediction available yet"  # If no prediction is available

# Call the training process
#train(epochs)  # Uncomment this line to start training
test(V_ten[-look_back:].tolist())
