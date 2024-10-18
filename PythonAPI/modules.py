import torch.nn as nn
import torch

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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