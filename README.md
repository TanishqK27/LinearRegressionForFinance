# Stock Price Prediction and Trading Strategy

Welcome to my personal project on stock price prediction and trading strategy using linear regression! This project is a culmination of my learning and application of linear regression techniques to predict stock prices and develop a trading strategy. The project is built using Python and leverages historical stock data to make future predictions. 

Bare in mind this was my first foray into the world of algorithmic trading/finance, hence it wasn't the greatest model in the world, but I did learn a lot. 

## Overview

In this project, I've implemented a linear regression model to predict the closing prices of stocks. Additionally, I've created a simple trading strategy based on these predictions to simulate buy and sell actions. The project also includes future price predictions for the next two months to demonstrate the model's capability.

## Features

- **Linear Regression Model**: Built from scratch using gradient descent to predict stock prices.
- **Technical Indicators**: Utilizes moving averages and RSI (Relative Strength Index) to enhance predictions and trading signals.
- **Trading Strategy**: Generates buy and sell signals based on predicted prices and evaluates the profit/loss of the strategy.
- **Future Price Prediction**: Simulates future stock prices for the next two months with added randomness to mimic market volatility.

## Project Structure

- **main.py**: The main script that runs the entire project.
- **data/**: Directory to store downloaded stock data.
- **plots/**: Directory to save generated plots of stock prices and trading signals.

## Example Output

Here are some example outputs generated by the project, all of which can be found in the plots directory:

### Stock Price Prediction
This plot shows the actual closing prices (blue line) and the predicted closing prices (orange dashed line) for Apple Inc. (AAPL). The model was trained on historical data from January 2020 to January 2023.

### Trading Signals
This plot highlights the buy (green upward arrows) and sell (red downward arrows) signals based on the predicted prices. The signals are generated using a threshold and the RSI indicator to enhance decision-making.

### Profit/Loss Evaluation
This plot displays the cumulative profit/loss over time based on the simulated trading strategy. The initial cash is $10,000, and the strategy buys one share when a buy signal is triggered and sells one share when a sell signal is triggered.



### Future Price Prediction
This plot shows the predicted stock prices for the next two months (approximately 60 trading days) from the last available date in the dataset. The future predictions include some randomness to simulate market volatility.

![Future Price Prediction](plots/future_price_prediction.png)

**Note:** This is a very simple project, and the profit/loss results are not optimized. More sophisticated models and strategies would be required for better performance in real-world trading.
