import torch
from modules import LSTM
import numpy as np


# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 1
hidden_size = 75
output_size = 1
num_layers = 1  # Number of LSTM layers

look_back = 24

model = LSTM(input_size, hidden_size, output_size, num_layers)

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
