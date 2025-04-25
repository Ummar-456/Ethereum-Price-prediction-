Ethereum Price Prediction with LSTM
This repository contains a deep learning model for predicting Ethereum (ETH) prices using Long Short-Term Memory (LSTM) neural networks. The model analyzes historical price data from Yahoo Finance to forecast future price movements.
Overview
The project implements a sequential LSTM model to predict cryptocurrency prices based on historical data. It uses a sliding window approach where the previous 60 data points (5-minute intervals) are used to predict the next price point.
Features
Data Retrieval: Fetches ETH-USD price data from Yahoo Finance
Preprocessing: Normalizes data and creates sequence-based training examples
Model Architecture: Implements a stacked LSTM model with dropout layers
Evaluation: Calculates prediction accuracy using Mean Squared Error (MSE)
Visualization: Plots training/validation loss and prediction results
Results Export: Saves predictions to CSV for further analysis
Model Architecture
The model consists of:
2 LSTM layers with 50 units each
Dropout layers (20% dropout rate) for regularization
Dense output layer
Results
For the most recent test run:
Mean Squared Error: 30.11
Data Points Processed: 8,869 sequences
Training Set Size: 5,676 sequences
Validation Set Size: 1,419 sequences
Test Set Size: 1,774 sequences
The model shows promising results with relatively small percentage differences between predicted and actual prices (typically less than 0.4%).

Visualizations
The script generates three visualizations:
Training and validation loss curves
Actual vs. predicted prices
Price differences between actual and predicted values
Limitations and Future Work
The model currently only uses closing prices; incorporating volume and other technical indicators could improve accuracy
Hyperparameter tuning could further optimize model performance
Implementing additional evaluation metrics beyond MSE
Testing different sequence lengths beyond 60 time steps
License
MIT License
