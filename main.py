import ccxt
import os
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Suppress TensorFlow oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure logging
logging.basicConfig(filename='eth_price_errors.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')


def fetch_coinbase_ohlcv(symbol="ETH/USD", timeframe="1d", start_date="2019-01-01", end_date="2023-12-31"):
    """
    Fetches historical OHLCV data spanning multiple years.

    Args:
        symbol (str): Trading pair symbol.
        timeframe (str): Timeframe for OHLCV data.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.

    Returns:
        pd.DataFrame: DataFrame containing OHLCV data.
    """
    exchange = ccxt.coinbaseexchange()
    since = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_timestamp = int(pd.Timestamp(end_date).timestamp() * 1000)

    all_data = []
    while since < end_timestamp:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=300)
            if not ohlcv:
                break
            data = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            since = ohlcv[-1][0] + 1  # Update `since` to avoid duplicate records
            all_data.append(data)
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            break

    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.sort_values("timestamp", inplace=True)
        return df
    else:
        raise ValueError("No data fetched from the exchange.")


def add_features(df):
    df.sort_values(by="timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["ma_5"] = df["close"].rolling(window=5).mean()
    df["ma_10"] = df["close"].rolling(window=10).mean()
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    df["macd"] = ta.trend.macd(df["close"])
    df["bb_hband"] = ta.volatility.BollingerBands(df["close"], window=20).bollinger_hband()
    df["bb_lband"] = ta.volatility.BollingerBands(df["close"], window=20).bollinger_lband()
    for lag in range(1, 4):
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
    df.dropna(inplace=True)
    return df


def train_xgboost(X_train, y_train):
    model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    return model


def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(50, return_sequences=True),
        Dropout(0.3),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='mae')
    return model


def detect_trends(residuals, window=3, slope_threshold=0.01):
    """Detects up and down trends in residuals based on rolling slope."""
    up_trends = []
    down_trends = []

    for i in range(len(residuals) - window + 1):
        x = np.arange(window)
        y = residuals[i:i + window]
        slope, _ = np.polyfit(x, y, 1)

        if slope > slope_threshold:
            up_trends.append(i + window - 1)
        elif slope < -slope_threshold:
            down_trends.append(i + window - 1)

    return up_trends, down_trends


def plot_combined_chart_with_future(df, y_test, y_pred_lstm, y_pred_xgb, residuals, up_trends, down_trends,
                                    future_predictions):
    """Plots Actual vs. Predicted Prices with Residual Trends and Future Predictions."""
    plt.figure(figsize=(14, 7))

    # Plot actual vs. predicted prices
    plt.plot(df.loc[y_test.index, "timestamp"], y_test, label="Actual", linewidth=2)
    plt.plot(df.loc[y_test.index, "timestamp"], y_pred_lstm, label="LSTM Predicted", linewidth=2, linestyle="--")
    plt.plot(df.loc[y_test.index, "timestamp"], y_pred_xgb, label="XGBoost Predicted", linewidth=2, linestyle="-.",
             color="green")

    # Highlight LSTM and XGBoost last predictions
    plt.scatter(df.loc[y_test.index[-1], "timestamp"], y_pred_lstm[-1], color="red", label="LSTM Prediction Marker",
                zorder=5)
    plt.scatter(df.loc[y_test.index[-1], "timestamp"], y_pred_xgb[-1], color="blue", label="XGBoost Prediction Marker",
                zorder=5)

    # Add future predictions to the plot
    future_dates = [pred[0] for pred in future_predictions]
    future_lstm = [pred[1] for pred in future_predictions]
    future_xgb = [pred[2] for pred in future_predictions]

    plt.plot(future_dates, future_lstm, label="Future LSTM Predictions", linestyle="--", marker="o", color="orange")
    plt.plot(future_dates, future_xgb, label="Future XGBoost Predictions", linestyle="-.", marker="o", color="purple")

    # Plot residuals and trends
    plt.plot(df.loc[y_test.index, "timestamp"], residuals, label="Residuals (Actual - Predicted)", color="blue",
             alpha=0.5)

    # Mark up trends
    plt.scatter(df.loc[y_test.index[up_trends], "timestamp"], np.array(residuals)[up_trends], color="green",
                label="Up Trend", zorder=5)
    plt.scatter(df.loc[y_test.index[down_trends], "timestamp"], np.array(residuals)[down_trends], color="red",
                label="Down Trend", zorder=5)

    # Adjust x-axis ticks
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Actual vs. Predicted Prices with Future Predictions (ETH)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def forecast_future(df, model_lstm, model_xgb, scaler, y_scaler, features, n_days=10):
    """Forecast future prices for LSTM and XGBoost."""
    future_predictions = []
    last_row = df.iloc[-1].copy()

    for i in range(n_days):
        # Prepare features for the next day
        feature_values = last_row[features].values.reshape(1, -1)
        feature_values_scaled = scaler.transform(feature_values)

        # LSTM Prediction
        feature_values_lstm = feature_values_scaled.reshape(
            (feature_values_scaled.shape[0], feature_values_scaled.shape[1], 1))
        next_pred_lstm_scaled = model_lstm.predict(feature_values_lstm).flatten()[0]
        next_pred_lstm = y_scaler.inverse_transform([[next_pred_lstm_scaled]])[0][0]

        # XGBoost Prediction
        next_pred_xgb = model_xgb.predict(feature_values_scaled)[0]

        # Calculate the date for the prediction
        next_date = last_row["timestamp"] + pd.Timedelta(days=1)
        last_row["timestamp"] = next_date

        # Save predictions with the corresponding date
        future_predictions.append((next_date.date(), next_pred_lstm, next_pred_xgb))

        # Update `last_row` for next iteration
        last_row["close"] = next_pred_lstm
        for lag in range(1, 4):
            last_row[f"close_lag_{lag}"] = last_row[f"close_lag_{lag - 1}"] if lag > 1 else next_pred_lstm
        last_row["ma_5"] = df["close"].iloc[-5:].mean()
        last_row["ma_10"] = df["close"].iloc[-10:].mean()
        last_row["rsi"] = ta.momentum.rsi(df["close"].iloc[-15:], window=14).iloc[-1]
        last_row["macd"] = ta.trend.macd(df["close"].iloc[-35:]).iloc[-1]
        last_row["bb_hband"] = ta.volatility.BollingerBands(df["close"].iloc[-20:], window=20).bollinger_hband().iloc[
            -1]
        last_row["bb_lband"] = ta.volatility.BollingerBands(df["close"].iloc[-20:], window=20).bollinger_lband().iloc[
            -1]

    return future_predictions


def main():
    try:
        df = fetch_coinbase_ohlcv(start_date="2019-01-01", end_date="2025-01-30")
        df = add_features(df)
        features = ["log_return", "ma_5", "ma_10", "rsi", "macd", "bb_hband", "bb_lband", "close_lag_1", "close_lag_2",
                    "close_lag_3"]
        target = "close"
        X = df[features]
        y = df[target]

        tscv = TimeSeriesSplit(n_splits=5)
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        y_scaler = MinMaxScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
        y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

        xgb_model = train_xgboost(X_train_scaled, y_train)
        y_pred_xgb = xgb_model.predict(X_test_scaled)

        mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
        print(f"XGBoost MAE: {mae_xgb:.3f}")

        X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
        X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

        lstm_model = build_lstm_model((X_train_lstm.shape[1], 1))
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lstm_model.fit(X_train_lstm, y_train_scaled, validation_split=0.2, epochs=100, batch_size=16, verbose=1,
                       callbacks=[early_stopping])
        y_pred_lstm = lstm_model.predict(X_test_lstm).flatten()

        y_pred_lstm_rescaled = y_scaler.inverse_transform(y_pred_lstm.reshape(-1, 1)).flatten()

        mae_lstm = mean_absolute_error(y_test, y_pred_lstm_rescaled)
        print(f"LSTM MAE: {mae_lstm:.3f}")

        # Print daily predictions
        print("\nDaily Predictions:")
        for i, timestamp in enumerate(df.loc[y_test.index, "timestamp"]):
            print(
                f"Date: {timestamp.date()}, LSTM Prediction: {y_pred_lstm_rescaled[i]:.2f}, XGBoost Prediction: {y_pred_xgb[i]:.2f}")

        # Calculate residuals
        residuals = y_test.values - y_pred_lstm_rescaled

        # Detect trends in residuals
        up_trends, down_trends = detect_trends(residuals)

        # Forecast future prices
        future_predictions = forecast_future(df, lstm_model, xgb_model, scaler, y_scaler, features, n_days=10)

        # Plot combined chart with future predictions
        plot_combined_chart_with_future(df, y_test, y_pred_lstm_rescaled, y_pred_xgb, residuals, up_trends, down_trends,
                                        future_predictions)

        # Print future predictions
        print("\nFuture Predictions:")
        for next_date, lstm_pred, xgb_pred in future_predictions:
            print(f"Date: {next_date}, LSTM Prediction: {lstm_pred:.2f}, XGBoost Prediction: {xgb_pred:.2f}")

        print(
            "\nNote: Your TensorFlow installation may not be optimized for AVX2/FMA. Consider recompiling TensorFlow for better performance.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print("An error occurred. Check logs for details.")


if __name__ == "__main__":
    main()
