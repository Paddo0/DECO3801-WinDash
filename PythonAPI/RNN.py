import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from torch.autograd import Variable

#Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#training data import 
df = pd.read_csv('PythonAPI\\data\\household_power_consumption.txt', sep=';')
#print(df["Voltage"])

# Normalization, using mean 0 var 1

df['Global_intensity'] = pd.to_numeric(df['Global_intensity'], errors='coerce')
df['Global_intensity'] = df['Global_intensity'].astype(float)
prev = df['Global_intensity'][::60]

scaler = MinMaxScaler(feature_range=(-1, 1))
values = prev[:100]
df['Global_intensity'] = scaler.fit_transform(df['Global_intensity'].values.reshape(-1,1))
V_ten = torch.FloatTensor(prev[:100].to_numpy()).view(-1).to(device)


# TODO Prepare the input and target sequences
# look-back period
look_back = 24

def create_inout_sequences(input_data, tw):
    "Creates input and output sequences out of a Tensor"
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
        self.lstm = nn.LSTM(input_size, hidden_size).to(device)
        self.linear = nn.Linear(hidden_size, output_size).to(device)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).to(device),
                            torch.zeros(1, 1, self.hidden_layer_size).to(device))


    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1).to(device), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1).to(device))
        return predictions[-1]


# TODO Tune hyperparameters 

input_size = 1
hidden_size = 100
output_size = 1



# Model initialization 
model = LSTM(input_size, hidden_size, output_size).to(device)

# TODO Selct appropriate loss criterion and optimizer 
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

# TODO determine sufficient number of epochs 
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

    print(f'Epoch {i} loss: {single_loss.item()}')

def test(inputs):
    model.load_state_dict(torch.load('model_weights.pth'))

    model.eval()

    for i in range(look_back):
        seq = torch.FloatTensor(inputs[-look_back:]).to(device)
        with torch.no_grad():
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                                torch.zeros(1, 1, model.hidden_layer_size).to(device))
            inputs.append(model(seq).item())

    # Inverse transform the predicted values to original scale
    predicted_values = scaler.inverse_transform(np.array(inputs[look_back:] ).reshape(-1, 1))
    print(predicted_values)

    # Plot the results
    import matplotlib.pyplot as plt

    plt.plot(scaler.inverse_transform(values.to_numpy().reshape(-1, 1)), label='Actual Data')
    plt.plot(np.arange(len(values), len(values) + look_back), predicted_values, label='Predictions')
    plt.legend()
    plt.show()
    plt.savefig('prediction.png')

    
def get_pred():
    # currently a vector of zeros as test 
    return [0,0,0,0,0]

train(epochs)
test(V_ten[-look_back:].tolist())