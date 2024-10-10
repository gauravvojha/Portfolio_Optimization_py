
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA

# Portfolio Risk Metrics
def sharpe_ratio(returns, risk_free_rate=0.01):
    return (returns.mean() - risk_free_rate) / returns.std()

def value_at_risk(returns, confidence_level=0.95):
    return np.percentile(returns, (1 - confidence_level) * 100)

# Data Fetching Function
def fetch_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

# Advanced ML Models: Random Forest and ARIMA for prediction
def train_ml_model(returns):
    X = returns.index.astype('int64').values.reshape(-1, 1)
    y = returns.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)
    
    # Predict using Random Forest
    future_dates = np.array([datetime.datetime(2024, 1, 1).toordinal()]).reshape(-1, 1)
    future_prediction_rf = rf_model.predict(future_dates)
    
    # Time Series prediction with ARIMA
    arima_model = ARIMA(returns, order=(5, 1, 0))
    arima_model_fit = arima_model.fit()
    future_prediction_arima = arima_model_fit.forecast(steps=10)[0]  # Predicting for next 10 periods
    
    return future_prediction_rf, future_prediction_arima

# Backtesting framework
def backtest_portfolio(returns, weights):
    portfolio_returns = (returns * weights).sum(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    return cumulative_returns

# Stress Testing with Historical Events
def stress_test(returns, event='2008_crisis'):
    if event == '2008_crisis':
        stress_factor = -0.35  # 35% drop in 2008 Financial Crisis
    elif event == 'covid_crash':
        stress_factor = -0.25  # 25% drop during COVID crash
    else:
        stress_factor = -0.2  # Default to 20% market decline
    stressed_returns = returns * (1 + stress_factor)
    return stressed_returns

# Main Function
def main():
    # User Inputs
    tickers = input("Enter Stock Tickers (comma-separated): ")
    start_date_str = input("Start Date (YYYY-MM-DD): ")
    end_date_str = input("End Date (YYYY-MM-DD): ")
    risk_tolerance = float(input("Risk Tolerance (0: Low, 1: High): "))
    esg_weighting = float(input("ESG Weighting (0: No ESG, 1: Full ESG): "))
    stress_scenario = input("Stress Test Scenario (2008_crisis, covid_crash, default): ")

    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').date()
    
    # Fetch and display stock data
    stock_data = fetch_data(tickers.split(', '), start_date, end_date)
    print("Stock Price Data:", stock_data.head())

    # Calculate returns
    returns = stock_data.pct_change().dropna()

    # Mock ESG Data
    esg_data = pd.DataFrame({
        'Ticker': ['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
        'ESG_Score': [85, 75, 90, 60]
    })

    # Combine ESG and returns data
    portfolio_data = pd.concat([returns.mean(), esg_data.set_index('Ticker')], axis=1).dropna()

    # Portfolio optimization function with ESG adjustment
    def optimize_portfolio(returns, esg_scores, risk_tolerance, esg_weighting):
        risk_adjusted_returns = returns / returns.std()
        combined_score = (1 - esg_weighting) * risk_adjusted_returns + esg_weighting * esg_scores
        weights = combined_score / combined_score.sum()  # Normalize weights
        return weights

    # Calculate optimal weights
    optimal_weights = optimize_portfolio(portfolio_data[0], portfolio_data['ESG_Score'], risk_tolerance, esg_weighting)
    print("Optimal Portfolio Weights:", optimal_weights)

    # Portfolio Risk Metrics
    portfolio_sharpe = sharpe_ratio(returns.mean())
    portfolio_var = value_at_risk(returns.mean())
    print("Sharpe Ratio:", portfolio_sharpe)
    print("Value at Risk (VaR):", portfolio_var)

    # Backtesting the portfolio
    cumulative_returns = backtest_portfolio(returns, optimal_weights)
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_returns, label='Cumulative Returns')
    plt.title('Portfolio Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

    # Stress Testing
    stressed_returns = stress_test(returns, event=stress_scenario)
    print(f"Portfolio Returns under {stress_scenario} scenario:", stressed_returns.head())

    # Machine Learning Predictions
    future_prediction_rf, future_prediction_arima = train_ml_model(stock_data['AAPL'])
    print("Random Forest Prediction for AAPL (2024):", future_prediction_rf)
    print("ARIMA Prediction for AAPL (next 10 periods):", future_prediction_arima)

    # Visualize original and stressed returns
    plt.figure(figsize=(12, 6))
    plt.plot(returns['AAPL'], label='Original Returns')
    plt.plot(stressed_returns['AAPL'], label=f'Returns under {stress_scenario}')
    plt.title('Original and Stressed Returns for AAPL')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
