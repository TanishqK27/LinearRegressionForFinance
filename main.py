import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import StandardScaler

# Fetch historical stock data
ticker = 'AAPL'
data = yf.download(ticker, start='2020-01-01', end='2023-01-01')

# Create features: Previous closing price, moving averages, and RSI
data['Previous Close'] = data['Close'].shift(1)
data['Moving Average 5'] = data['Close'].rolling(window=5).mean()
data['Moving Average 10'] = data['Close'].rolling(window=10).mean()

# RSI calculation
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# Drop the rows with missing values
data = data.dropna()

# Prepare the data
features = ['Previous Close', 'Moving Average 5', 'Moving Average 10', 'RSI']
target = 'Close'

X = data[features].values
y = data[target].values

# Split the data into training and testing sets
split_index = int(len(data) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add a column of ones for the intercept term
X_train_b = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
X_test_b = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]


# Mean Squared Error function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Gradient Descent function for multiple features
def gradient_descent_multi(X, y, learning_rate=0.01, num_iterations=1000):
    m, n = X.shape
    theta = np.zeros(n)  # Initial parameters
    for iteration in range(num_iterations):
        y_pred = X.dot(theta)  # Predicted y values
        errors = y_pred - y  # Errors
        gradient = (2 / m) * X.T.dot(errors)  # Compute gradient
        theta -= learning_rate * gradient  # Update parameters

        # Debug: Print the cost function value
        cost = mean_squared_error(y, y_pred)
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Cost {cost}")

    return theta


# Train the model
learning_rate = 0.01
num_iterations = 1000
theta = gradient_descent_multi(X_train_b, y_train, learning_rate, num_iterations)

print(f"Estimated parameters: {theta}")

# Make predictions
y_pred = X_test_b.dot(theta)


# Trading strategy based on predicted price with threshold and RSI
def trading_strategy(predictions, actual_prices, threshold=0.01, rsi=None):
    signals = []
    for i in range(1, len(predictions)):
        if predictions[i] > actual_prices[i - 1] * (1 + threshold) and (rsi is None or rsi[i] < 70):
            signals.append('Buy')
        elif predictions[i] < actual_prices[i - 1] * (1 - threshold) and (rsi is None or rsi[i] > 30):
            signals.append('Sell')
        else:
            signals.append('Hold')
    return signals


# Generate trading signals
signals = trading_strategy(y_pred, y_test, threshold=0.02, rsi=data['RSI'].values[split_index:])

# Add signals to the DataFrame for visualization
results = pd.DataFrame(data.index[split_index:], columns=['Date'])
results['Actual Price'] = y_test
results['Predicted Price'] = y_pred
results['Signal'] = ['Hold'] + signals  # First day has no signal

# Calculate profit/loss
initial_cash = 10000  # Initial cash in dollars
cash = initial_cash
stock_holding = 0
profits = []

for i in range(1, len(results)):
    if results['Signal'][i] == 'Buy':
        if cash >= results['Actual Price'][i]:  # Buy one share
            stock_holding += 1
            cash -= results['Actual Price'][i]
    elif results['Signal'][i] == 'Sell':
        if stock_holding > 0:  # Sell one share
            stock_holding -= 1
            cash += results['Actual Price'][i]
    total_value = cash + stock_holding * results['Actual Price'][i]
    profits.append(total_value - initial_cash)

# Add profits to results DataFrame
results['Profit/Loss'] = [0] + profits

# Print the first few rows of the results
print(results.head())

# Visualization
plt.figure(figsize=(14, 7))
plt.plot(results['Date'], results['Actual Price'], label='Actual Prices')
plt.plot(results['Date'], results['Predicted Price'], label='Predicted Prices', linestyle='--')

# Highlight buy/sell signals
buy_signals = results[results['Signal'] == 'Buy']
sell_signals = results[results['Signal'] == 'Sell']
plt.scatter(buy_signals['Date'], buy_signals['Actual Price'], marker='^', color='g', label='Buy Signal', alpha=1)
plt.scatter(sell_signals['Date'], sell_signals['Actual Price'], marker='v', color='r', label='Sell Signal', alpha=1)

plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'{ticker} Stock Price Prediction with Trading Signals')
plt.legend()
plt.show()

# Plot profit/loss over time
plt.figure(figsize=(14, 7))
plt.plot(results['Date'], results['Profit/Loss'], label='Profit/Loss')
plt.xlabel('Date')
plt.ylabel('Profit/Loss')
plt.title(f'{ticker} Trading Strategy Profit/Loss')
plt.legend()
plt.show()

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Predict future prices for the next 2 months (approx. 60 trading days)
future_dates = pd.date_range(start=data.index[-1], periods=60, freq='B')  # 'B' for business days
last_row = data.iloc[-1][features].values.reshape(1, -1)
future_predictions = []

for _ in range(60):
    last_row_scaled = scaler.transform(last_row)
    last_row_b = np.c_[np.ones((last_row_scaled.shape[0], 1)), last_row_scaled]
    next_price = last_row_b.dot(theta)
    future_predictions.append(next_price[0])

    # Add some randomness to future price predictions to simulate market volatility
    next_price[0] += np.random.normal(0, 2)

    # Update the last_row with the new predicted price and recalculate moving averages
    next_row = [next_price[0]] + list(last_row[0][1:])
    last_row = np.array(next_row).reshape(1, -1)
    last_row[0][1] = np.mean(future_predictions[-5:])  # 5-day moving average
    last_row[0][2] = np.mean(future_predictions[-10:])  # 10-day moving average
    last_row[0][3] = 100 - (100 / (1 + (gain[-1] / loss[-1])))  # RSI update

# Plot future price predictions
plt.figure(figsize=(14, 7))
plt.plot(future_dates, future_predictions, label='Future Predictions', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Predicted Closing Price')
plt.title(f'{ticker} Future Stock Price Prediction for Next 2 Months')
plt.legend()
plt.show()