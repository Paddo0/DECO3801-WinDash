
# DECO3801 - PythonAPI

## Summary

The purpose of this feature is to write a python script that contains a fully trained neural network for sequence predictions of time series power usage data. The API will retrieve the current usage data from the database, this will then be input into the model to genrate predictions to be posted back to the website. The model itself will be trained on real data gathered from Australian homes to learn the patterns of energy consumption. 
---

## API Functions

Flask API has been chosen for this feature as it provides a lightweight yet effective solution. Additionally, Flask has good integration with FireBase - our chosen database. The get request will retreive the lastest few points of time series data and the post method will output the prediction made by the neural net. 
---

## Models

The chosen model for this project is a vanailla RNN. RNN was the clear choice for predicting next time series data as the function of an RNN is to to process and convert a sequential data input into a specific sequential data output. Specifically, a vanilla variant was selected due to the limited amount of data available - as more advanced variants such as LSTM require an abundance of training data. Training techniques such as k-fold cross validation and boostrapping will be employed to maximise the infomation extracted from the data. The most significant part of this project will be tuning the architecture of the RNN and the hyperparamters - most significantly the loss criterion, optimizer, layer width and depth, learning rate and number of epochs. The tuning process will require extensive re-running of the model and potentially a grid search to find an effective solution. 
---