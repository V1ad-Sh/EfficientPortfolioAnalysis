import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Fetch Historical Stock Data
stocks = [
    '^GSPC', '^IXIC', '^RUT', '^DJI', '^NDX','ABNB', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'AMAT', 'AMD', 'AMGN', 'ANSS', 'ASML', 'AVGO', 'AZN', 'BIIB', 'BKNG', 'BKR', 'CCEP', 'CDNS', 'CDW', 'CHTR', 'CMCSA', 'COST', 'CPRT', 'CRWD', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTSH', 'DASH', 'DDOG', 'DLTR', 'DXCM', 'EA', 'EXC', 'FANG', 'FAST', 'FTNT','GOOGL', 'HON', 'IDXX', 'ILMN', 'INTC', 'INTU', 'ISRG', 'KDP', 'KHC', 'KLAC', 'LRCX', 'LULU', 'MAR', 'MCHP', 'MDB', 'MDLZ', 'MELI', 'META', 'MNST', 'MRNA', 'MRVL','MSFT', 'MU', 'NFLX', 'NVDA', 'NXPI', 'ODFL', 'ON', 'ORLY', 'PANW', 'PAYX', 'PCAR', 'PDD', 'PEP', 'PYPL', 'QCOM', 'REGN', 'ROP', 'ROST', 'SBUX', 'SIRI', 'SNPS', 'TEAM', 'TMUS', 'TSLA', 'TTD', 'TTWO', 'TXN', 'VRSK','VRTX', 'WBA', 'WBD', 'WDAY', 'XEL', 'ZS', 'GILD', 'GOOG', 'AMZN', 'AAPL',
]

start_date = '2021-03-01'
end_date = '2024-03-01'
data = yf.download(stocks, start=start_date, end=end_date)['Adj Close']

bonds = ['^IRX']
annual_bond_rates_in_percents = yf.download(bonds, start=start_date, end=end_date)["Adj Close"]
# de-annualize
annual_tbill_rates = annual_bond_rates_in_percents/100
def geometric_mean_daily_rate(annual_tbill_rates):
    # Convert the annual rates to a numpy array for vectorized operations
    annual_tbill_rates = np.array(annual_tbill_rates)
    d_year = 252  # Number of trading days in the year
    # Convert annual rates to daily rates
    daily_tbill_rates = (1 + annual_tbill_rates)**(1/d_year) - 1
    # Calculate the geometric mean of the daily rates
    # Use the numpy function for geometric mean
    geo_mean = np.prod(1 + daily_tbill_rates)**(1/daily_tbill_rates.size) - 1
    return geo_mean

# Calculate Expected Returns and Covariance Matrix
returns = data.pct_change().dropna()
log_returns = np.log(data / data.shift(1)).dropna()
cov_matrix = log_returns.cov()
cov_matrix = cov_matrix.reindex(index=stocks, columns=stocks)
geomean_returns = returns.apply(lambda x: np.prod(1 + x)**(1/len(x)) - 1)
geomean_returns = geomean_returns.reindex(stocks)
geomean_returns['Risk_Free'] = geometric_mean_daily_rate(annual_tbill_rates)
# Calculate Portfolio Risk
def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights[:-1].T, np.dot(cov_matrix, weights[:-1])))
# Calculate Portfolio Return
def portfolio_expected_return(weights, geomean_returns):
    return np.dot(weights, geomean_returns)
# Set Return Constraint
def portfolio_return_constraint(weights, geomean_returns, target_return):
    return np.dot(weights, geomean_returns) - target_return
# Define Objective Function
def minimize_volatility(geomean_returns, cov_matrix):
    num_assets = len(geomean_returns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_guess = num_assets * [1. / num_assets,]
    result = minimize(portfolio_volatility, initial_guess, args=(cov_matrix,), method='SLSQP', bounds=bounds, constraints=constraints)
    return result
# Range of target returns
min_return = np.dot(minimize_volatility(geomean_returns, cov_matrix).x,geomean_returns)
max_return = geomean_returns.max()
num_portfolios = 10
step = (max_return - min_return) / num_portfolios
# Optimization process
efficient_frontier = []
expected_returns = []  # List to store expected returns
volatilities = []      # List to store volatilities
for i in range(num_portfolios + 1):
    target_return = min_return + step * i
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                   {'type': 'eq', 'fun': lambda x: portfolio_return_constraint(x, geomean_returns, target_return)}]
    bounds = tuple((0, 1) for asset in geomean_returns)

    result = minimize(portfolio_volatility, [1. / len(geomean_returns)] * len(geomean_returns), args=(cov_matrix,), method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        weights = result.x
        expected_return = portfolio_expected_return(weights, geomean_returns)
        volatility = result.fun
        efficient_frontier.append((weights, expected_return, volatility))
        expected_returns.append(expected_return)#Append expected return to the list
        volatilities.append(volatility)  # Append volatility to the list

# Display the results and calculate Sharpe Ratios
sharpe_ratios = []

# Display the results
for i, (weights, expected_return, volatility) in enumerate(efficient_frontier):
    # Calculate Sharpe Ratio (Expected Return - Daily Risk-Free Rate) / Volatility
    sharpe_ratio = (expected_return - geomean_returns['Risk_Free']) / volatility if volatility != 0 else float('inf')
    sharpe_ratios.append(sharpe_ratio)
    
    # Separate weights for stocks and the risk-free asset
    stock_weights = weights[:-1]  # all except the last element
    bond_weight = weights[-1]  # the last element
    
    # Combine stock names with their corresponding weights
    named_weights = [(stock, weight) for stock, weight in zip(stocks, stock_weights) if weight > 0.00001]
    
    # Format the stock weights
    formatted_stock_weights = [f"{idx + 1}. {stock}: {weight * 100:.2f}%" for idx, (stock, weight) in
                               enumerate(named_weights)]
    
    # Format the bond (risk-free asset) weight
    formatted_bond_weight = f"Bond (Risk-Free): {bond_weight * 100:.2f}%"
    
    print(f"Portfolio {i + 1}:")
    print("Weights:", ' '.join(formatted_stock_weights), formatted_bond_weight)
    print(f"Expected Return: {expected_return:.4%}")
    print(f"Volatility: {volatility:.4%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print()
    
# Plotting the Efficient Frontier
plt.figure(figsize=(10, 6))
plt.plot(volatilities, expected_returns, marker='o') # 'o' creates a circle marker at each data point
plt.title('Efficient Frontier')
plt.xlabel('Volatility (Standard Deviation)')
plt.ylabel('Expected Return')
plt.grid(True)
plt.show()