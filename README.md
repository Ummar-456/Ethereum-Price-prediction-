Ethereum Price Prediction with LSTM Neural Networks
This repository implements a deep learning solution for predicting Ethereum (ETH) cryptocurrency prices using Long Short-Term Memory (LSTM) neural networks. The model analyzes historical price data at 5-minute intervals to forecast short-term price movements, providing valuable insights for traders and investors in the volatile cryptocurrency market.

# Table of Contents
1- Features.
2- Technical Architecture
Data Pipeline
Model Architecture
Performance Metrics
Sample Results
Installation
Usage
Visualizations
Limitations
Future Improvements
Contributing
License

# Features
Real-time Data Retrieval: Fetches the latest ETH-USD price data from Yahoo Finance API
Automated Data Preprocessing: Normalizes and sequences data for optimal model training
Advanced LSTM Architecture: Implements a stacked LSTM model with dropout for regularization
Comprehensive Evaluation: Calculates prediction accuracy using Mean Squared Error (MSE)
Detailed Visualization: Generates plots for model performance and prediction accuracy
Results Export: Saves predictions to CSV for integration with trading systems

# **Technical Architecture**
The project follows a modular architecture with the following components:
Data Acquisition Module: Fetches historical price data using the Yahoo Finance API
Preprocessing Module: Normalizes data and creates sequence-based training examples
Model Definition Module: Implements the LSTM neural network architecture
Training Module: Handles model training with validation
Prediction Module: Generates price forecasts
Evaluation Module: Calculates performance metrics
Visualization Module: Creates informative plots
Data Pipeline
The data pipeline consists of the following steps:
Data Collection: Retrieves ETH-USD price data at 5-minute intervals for the past month
Data Cleaning: Handles missing values and ensures data integrity
Feature Scaling: Normalizes price data to the range using MinMaxScaler
Sequence Creation: Transforms data into sliding windows of 60 time steps
Train-Validation-Test Split: Divides data into training (64%), validation (16%), and test (20%) sets

**Performance Metrics**
**For the most recent test run:**
**Mean Squared Error:** 30.11
**Root Mean Squared Error:**	5.49
**Average Percentage Difference:**	0.34%
**Training Time:**	~2-3 minutes
**Data Points Processed:**	8,869 sequences
**Training Set Size:**	5,676 sequences
**Validation Set Size:**	1,419 sequences
**Test Set Size:** 1,774 sequences


# Usage
Run the main script to fetch data, train the model, and generate predictions:

python eth_price_prediction.py

# Customization Options
You can customize the model by modifying the following parameters in the script:
**symbol:** Change to predict different cryptocurrencies (e.g., 'BTC-USD')
**interval:** Adjust the time interval (e.g., '1h' for hourly data)
**period:** Modify the historical data period (e.g., '3mo' for 3 months)
**sequence_length:** Change the number of time steps used for prediction (default: 60)

# Visualizations
The script generates three key visualizations:
1. Model Loss: Model Loss and validation loss over epochs, helping to identify overfitting or underfitting.
2. Predictions vs Actual Values
3. Price Differences

# Limitations
**Short-term Focus:** The model is designed for short-term predictions and may not capture long-term trends.
**Limited Features:** Currently only uses closing prices without additional technical indicators.
**Market Sensitivity:** Cryptocurrency markets are highly volatile and can be influenced by external factors not captured in historical data.
**Data Availability:** Relies on the quality and availability of data from Yahoo Finance.
**Fixed Sequence Length:** Uses a fixed window of 60 time steps which may not be optimal for all market conditions.

**Future Improvements**
**Feature Engineering:** Incorporate additional features such as volume etc.
**Hyperparameter Optimization:**  Implement grid search or Bayesian optimization for hyperparameter tuning.
**Ensemble Methods:** Combine predictions from multiple models for improved accuracy.
**Attention Mechanisms:** Implement attention layers to focus on the most relevant time steps.
**Sentiment Analysis: ** Integrate social media sentiment data to capture market psychology.
**Real-time Predictions:** Develop a real-time prediction system with automated trading signals.
**Transfer Learning:** Pre-train on related cryptocurrencies to improve performance.

# Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
# Fork the repository
Create your feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add some amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request
# License
This project is licensed under the MIT License - see the LICENSE file for details.
Disclaimer: This project is for educational purposes only. Cryptocurrency trading involves significant risk, and past performance is not indicative of future results. Always conduct your own research before making investment decisions.
# Clone the repository:
git clone https://github.com/yourusername/eth-price-prediction.git
cd eth-price-prediction


# Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  #On Windows: venv\Scripts\activate

# Install dependencies:
pip install -r requirements.txt
