
# DECO3801 - PythonAPI

## Summary

This feature is designed to implement a Python API that integrates a neural network for time series predictions on power usage data. The API retrieves historical usage data from the database (FireStore), processes it through the trained model, and generates predictions, which are then sent back to the front-end for visualization. The data used for training and predictions is gathered from real-world Australian households, enabling the model to learn the energy consumption patterns for both daily and monthly predictions.
---

## API Functions

Flask was chosen as the framework for this API due to its simplicity, lightweight nature, and easy integration with Firebase, our cloud database. The API features two main routes:
    1. Daily Prediction: Fetches daily power consumption data from FireStore and uses the trained model to predict the next period's consumption.
    2. Monthly Prediction: Fetches monthly aggregated consumption data and generates predictions for future month.
Both endpoints retrieve the latest available data points via POST requests and feed them into the RNN model, which then outputs the predictions in real-time. The architecture allows for easy scaling and customization, enabling future enhancements if needed.


---

## Models

The core model utilized for this project is a Recurrent Neural Network (RNN). After evaluating multiple architectures, the decision was made to go with an RNN due to its capability to effectively handle sequential data, such as time series power usage, while requiring fewer resources compared to more advanced models like LSTMs.
The RNN processes the input data (time series of power usage values) and is optimized for:
    1. Daily Data: Uses the most recent 60-minute average values to generate short-term predictions for daily power consumption.
    2. Monthly Data: Aggregated monthly data, such as average intensity and total consumption, is used to generate predictions for future months.
These models have been tuned to work with limited training data by focusing on simpler architectures that don’t require vast amounts of data. Techniques such as averaging data points and leveraging sequential patterns have been employed to improve prediction accuracy.

Training Process：
    1. Data Preprocessing: The data is first cleaned, with missing values handled appropriately. For daily predictions, we use the average of every 60 minutes to reduce noise and enhance the model’s ability to learn from trends.
    2. Training Strategy: The RNN is trained on batches of time-series data, using sequences of 60 input values to predict future consumption. The training includes multiple rounds of hyperparameter tuning, focusing on finding the optimal learning rate, loss function, and model architecture.
    3. Evaluation: The model's performance is evaluated using metrics like Mean Squared Error (MSE) and Mean Absolute Error (MAE) to ensure it generalizes well to new data.

---