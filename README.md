# Portfolio_Optimization_py

# Portfolio Optimization & Prediction with Machine Learning and ESG Integration

This Python project demonstrates an advanced portfolio optimization model, integrating ESG (Environmental, Social, and Governance) data into the investment strategy, along with risk metrics, backtesting, stress testing, and machine learning predictions using Random Forest and ARIMA models.
Key Features

    Stock Data Fetching: Download stock data using the yfinance API.
    Risk Metrics: Calculate key portfolio risk metrics such as the Sharpe Ratio and Value at Risk (VaR).
    Portfolio Optimization with ESG: Optimize the portfolio based on stock returns and ESG scores.
    Backtesting: Evaluate portfolio performance over time using historical data.
    Stress Testing: Simulate portfolio performance under historical market stress events like the 2008 financial crisis and COVID crash.
    Machine Learning Predictions: Use Random Forest and ARIMA models to predict stock prices.

Table of Contents

    Installation
    Usage
    Portfolio Risk Metrics
    Data Fetching
    ESG-based Portfolio Optimization
    Backtesting
    Stress Testing
    Machine Learning Predictions
    Visualization

# Installation

To run this project, you need the following Python libraries:

bash

pip install numpy pandas matplotlib seaborn yfinance scikit-learn statsmodels

Usage

You can execute the program by running the main script, which prompts user input for stock tickers, start and end dates, and risk tolerance levels. The system will fetch stock data, calculate portfolio weights, and provide insights into portfolio performance under normal and stressed market conditions.

bash

python portfolio_optimization.py

Inputs:

    Stock Tickers: Enter the tickers of stocks, comma-separated (e.g., AAPL, GOOGL, TSLA).
    Start Date: The start date for fetching historical stock data (format: YYYY-MM-DD).
    End Date: The end date for fetching historical stock data (format: YYYY-MM-DD).
    Risk Tolerance: Value between 0 (low risk) and 1 (high risk).
    ESG Weighting: Value between 0 (no ESG consideration) and 1 (full ESG weighting).
    Stress Test Scenario: Select stress scenario for the portfolio simulation (options: 2008_crisis, covid_crash, default).

# Portfolio Risk Metrics

    Sharpe Ratio: Measures the risk-adjusted return of the portfolio. Higher Sharpe ratios indicate better risk-adjusted returns.

    python

sharpe_ratio(returns)

Value at Risk (VaR): Estimates the maximum expected loss over a specific time period with a given confidence level (e.g., 95%).

python

    value_at_risk(returns, confidence_level=0.95)

# Data Fetching

Stock price data is fetched from Yahoo Finance using the yfinance library, specifically the Adjusted Close price to account for dividends and stock splits.

python

fetch_data(tickers, start_date, end_date)

# ESG-based Portfolio Optimization

This section combines stock returns with ESG scores, adjusting weights based on a user-defined ESG weighting factor. The portfolio is optimized to balance risk tolerance and ESG considerations.

python

optimize_portfolio(returns, esg_scores, risk_tolerance, esg_weighting)

# Backtesting

Backtest the performance of the optimized portfolio by simulating how it would have performed historically using cumulative returns.

python

backtest_portfolio(returns, weights)

# Stress Testing

Simulate the performance of the portfolio under historical market stress events, such as the 2008 financial crisis or the COVID crash.

python

stress_test(returns, event='2008_crisis')

Supported Scenarios:

    2008_crisis: 35% drop in market value.
    covid_crash: 25% drop in market value.

# Machine Learning Predictions

The project uses two machine learning models for predicting stock prices:

    Random Forest: Predicts future stock prices using a Random Forest Regressor model.
    ARIMA (AutoRegressive Integrated Moving Average): A time series forecasting model used to predict future stock returns.

python

train_ml_model(stock_data['AAPL'])

# Predictions:

    Random Forest: Predicts stock prices on a future date (e.g., January 1, 2024).
    ARIMA: Predicts stock returns for the next 10 time periods.

# Visualization

The project provides visual insights, including:

    Cumulative Returns: A plot showing the cumulative returns of the optimized portfolio.
    Stress Test Comparison: A comparison between original and stressed returns.

python

plt.plot(cumulative_returns)

# Example

After running the script, the system will display:

    Optimal Portfolio Weights based on stock returns and ESG scores.
    Portfolio Risk Metrics including the Sharpe Ratio and Value at Risk.
    Cumulative Returns Plot showing the portfolio's performance over time.
    Stress Test Results indicating portfolio performance during extreme market conditions.
    Machine Learning Predictions for future stock prices using Random Forest and ARIMA models.

# License

This project is licensed under the MIT License.

By incorporating ESG data into your portfolio, combined with machine learning predictions and stress testing, this project enables an advanced investment strategy tailored to both financial returns and sustainability.
