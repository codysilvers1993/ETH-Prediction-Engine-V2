🚀 ETH/USD Prediction Engine

📌 Overview
This project is an Ethereum (ETH/USD) price prediction engine that leverages machine learning models, specifically:

LSTM (Long Short-Term Memory)

XGBoost (Extreme Gradient Boosting)

It provides daily price forecasts and detects market trend adherence, marking key points with red/green indicators for easy interpretation.

✨ Features

Price Prediction

Uses LSTM and XGBoost models for highly accurate forecasting.
Provides daily and future predictions.

📈 Trend Detection

Red Dots 🔴 - Adherence to the model's predicted trend.

Green Dots 🟢 - Non-adherence to the model's predicted trend.

Technical Indicators Used

Moving Averages (MA5, MA10)

RSI (Relative Strength Index)

MACD (Moving Average Convergence Divergence)

Bollinger Bands (BB Upper/Lower)

Advanced Analysis

Detects uptrends and downtrends in residuals.

Implements rolling window slope detection to highlight market shifts.

Future Forecasting

Predicts up to 10 days ahead for both LSTM and XGBoost.

Generates a visualized price trend chart.

Backtesting Support

Backtests models with up to 5 years of historical data fetched from Coinbase Pro API.

Uses TimeSeriesSplit for validation.

To utilize model for editing/open source collab please clone the repo and install dependencies listed in the .txt file.
