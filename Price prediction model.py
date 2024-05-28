

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Input
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

def fetch_data(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """
    Fetch historical market data for a given symbol from Yahoo Finance.

    Args:
        symbol (str): Ticker symbol of the asset.
        interval (str): Data interval (e.g., '5m' for 5 minutes).
        period (str): Time period (e.g., '1mo' for 1 month).

    Returns:
        pd.DataFrame: DataFrame containing the fetched data.
    """
    return yf.download(tickers=symbol, interval=interval, period=period)

def preprocess_data(df: pd.DataFrame) -> tuple:
    """
    Preprocess the data for LSTM model training.

    Args:
        df (pd.DataFrame): DataFrame containing the market data.

    Returns:
        tuple: Preprocessed input features, target values, and scaler.
    """
    if 'Close' not in df.columns or df['Close'].isnull().values.any():
        raise ValueError("DataFrame must contain a 'Close' column with no missing values.")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    
    sequence_length = 60  # Use last 60 minutes to predict the next price
    x, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        x.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    x = np.array(x)
    y = np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    
    return x, y, scaler

def build_and_train_model(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> tuple:
    """
    Build and train the LSTM model.

    Args:
        x_train (np.ndarray): Training input features.
        y_train (np.ndarray): Training target values.
        x_val (np.ndarray): Validation input features.
        y_val (np.ndarray): Validation target values.

    Returns:
        tuple: Trained model and training history.
    """
    model = Sequential([
        Input(shape=(x_train.shape[1], 1)),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_val, y_val))
    
    return model, history

def make_predictions(model: Sequential, x_test: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """
    Make predictions using the trained LSTM model.

    Args:
        model (Sequential): Trained LSTM model.
        x_test (np.ndarray): Test input features.
        scaler (MinMaxScaler): Fitted scaler to inverse transform the predictions.

    Returns:
        np.ndarray: Inverse transformed predictions.
    """
    predictions = model.predict(x_test)
    return scaler.inverse_transform(predictions)

if __name__ == "__main__":
    symbol = 'ETH-USD'
    interval = '5m'
    period = '1mo'

    print("Fetching data from Yahoo Finance...")
    df = fetch_data(symbol, interval, period)
    print("Data fetched successfully.")
    print(df.head(2))  # Display the first few rows of the data

    x, y, scaler = preprocess_data(df)
    print("Data preprocessed successfully.")
    print(f"x shape: {x.shape}, y shape: {y.shape}")  # Display the shape of the data

    # Splitting data into training, validation, and test sets
    split = int(0.8 * len(x))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    val_split = int(0.8 * len(x_train))
    x_val, y_val = x_train[val_split:], y_train[val_split:]
    x_train, y_train = x_train[:val_split], y_train[:val_split]

    print("Building and training LSTM model...")
    model, history = build_and_train_model(x_train, y_train, x_val, y_val)
    print("Model trained successfully.")

    predictions = make_predictions(model, x_test, scaler)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    mse = mean_squared_error(y_test_rescaled, predictions)
    print(f"Mean Squared Error on test data: {mse}")

    # Calculate price differences
    differences = y_test_rescaled.flatten() - predictions.flatten()
    percentage_differences = (differences / y_test_rescaled.flatten()) * 100

    # Add timestamps
    timestamps = df.index[-len(y_test):]

    # Create a DataFrame with results
    results_df = pd.DataFrame({
        'Timestamp': timestamps,
        'Actual Prices': y_test_rescaled.flatten(),
        'Predicted Prices': predictions.flatten(),
        'Differences': differences,
        'Percentage Differences': percentage_differences
    })

    # Save to CSV
    results_df.to_csv('predictions_vs_actuals.csv', index=False)

    # Print confirmation and first few rows
    print("Results saved to predictions_vs_actuals.csv")
    print(results_df.head())

    # Plot validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot predictions vs actual values
    plt.figure(figsize=(10, 5))
    plt.plot(results_df['Timestamp'], results_df['Actual Prices'], color='blue', label='Actual Values')
    plt.plot(results_df['Timestamp'], results_df['Predicted Prices'], color='red', label='Predicted Values')
    plt.title('Predictions vs Actual Values')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Plot price differences
    plt.figure(figsize=(10, 5))
    plt.plot(results_df['Timestamp'], results_df['Differences'], color='green', label='Price Differences')
    plt.title('Price Differences: Actual vs Predicted')
    plt.xlabel('Time')
    plt.ylabel('Difference')
    plt.legend()
    plt.show()
