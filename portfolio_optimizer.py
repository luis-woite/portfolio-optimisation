#   1. Download daily prices for several stocks
#   2. Compute daily % returns
#   3. Calculate annualized return & volatility
#   4. Build covariance and correlation matrices

import yfinance as yf
import pandas as pd
import numpy as np

from scipy.optimize import minimize


# Choose which stocks we want 
tickers = ["AAPL", "MSFT", "META", "GOOG", "XOM"]

# Download daily closing prices
data = yf.download(tickers,
                   start="2024-01-01",
                   end="2024-12-31",
                   progress=False)["Close"]

# Clean the data
data = data.dropna()

# ---------- 5) Take a quick look ----------
print(" First few rows of daily closing prices:")
print(data.head())

# Compute daily percentage returns

daily_returns = data.pct_change().dropna()

print("\n✅ First few rows of daily returns (in decimals):")
print(daily_returns.head())

# Annualize mean returns and volatility
# Convention: there are roughly 252 trading days per year.


# Annualized Return ≈ mean_daily_return × 252
# Annualized Volatility ≈ std_daily_return × √252
# ============================================================

trading_days = 252

# mean() computes the average daily return for each stock
mean_daily = daily_returns.mean()

# Convert daily metrics to annual metrics
annual_returns = mean_daily * trading_days
annual_vol = daily_returns.std() * np.sqrt(trading_days)

print("\n Annualized Expected Returns (per stock):")
print(annual_returns)

print("\n Annualized Volatilities (per stock):")
print(annual_vol)


# Covariance and Correlation matrices

cov_matrix = daily_returns.cov()
corr_matrix = daily_returns.corr()



print("\n Covariance matrix (daily):")
print(cov_matrix)

print("\n Correlation matrix:")
print(corr_matrix)




# Portfolio return and risk for a given set of weights



# choose some weights

weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

# quick safety check
if not np.isclose(weights.sum(), 1.0):
    raise ValueError("Weights must sum to 1.0!")

# portfolio annual return 

portfolio_return = np.dot(weights, annual_returns)

# portfolio annual volatility 
# cov_matrix is currently DAILY covariance.
# To get ANNUAL covariance, multiply by number of trading days.
annual_cov_matrix = cov_matrix * trading_days

# Formula: σ_p = sqrt( w' Σ w )
portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(annual_cov_matrix, weights)))

print("\n===== Portfolio (equal weights) =====")
print(f"Weights: {dict(zip(tickers, weights))}")
print(f"Expected annual return: {portfolio_return:.2%}")
print(f"Expected annual volatility: {portfolio_vol:.2%}")



# Helper functions for portfolio math

def portfolio_return(weights, annual_returns):
    """
    Calculate portfolio expected return.
    weights: 1D numpy array (length = number of assets)
    annual_returns: pandas Series with annual return per asset
    """
    return np.dot(weights, annual_returns)


"""
computes the portfolio weighted sum of annual returns

"""



def portfolio_vol(weights, annual_cov_matrix):
    """
    Calculate portfolio volatility (standard deviation).
    Formula: sqrt( w' * Σ * w )
    """
    return np.sqrt(np.dot(weights.T, np.dot(annual_cov_matrix, weights)))

"""
standard formula for portfolio volatility
"""




"""
note: np.dot returns dot porduct of two arrays.

np.dot([a, b], [c, d]) = a*c + b*d


"""

# Find the minimum-volatility portfolio

num_assets = len(tickers)

# starting guess: equal weights
x0 = np.array([1/num_assets] * num_assets)

# constraint 1: weights must sum to 1
constraints = ({
    "type": "eq",               # equality constraint
    "fun": lambda w: np.sum(w) - 1
})

# bounds: each weight between 0 and 1
bounds = tuple((0, 1) for _ in range(num_assets))

# run the optimizer
result = minimize(
    fun=portfolio_vol,              # objective: minimize volatility
    x0=x0,                          # initial guess
    args=(annual_cov_matrix,),      # extra args passed to portfolio_vol
    method="SLSQP",                 # good for constrained problems
    bounds=bounds,
    constraints=constraints
)

# checking if it worked
if not result.success:
    print("Optimization failed:", result.message)

opt_weights = result.x



# calculate stats for the optimal portfolio
opt_return = portfolio_return(opt_weights, annual_returns)
opt_vol = portfolio_vol(opt_weights, annual_cov_matrix)

print("\n===== Minimum-Volatility Portfolio =====")
for ticker, w in zip(tickers, opt_weights):
    print(f"{ticker}: {w:.2%}")
print(f"Portfolio expected return: {opt_return:.2%}")
print(f"Portfolio volatility    : {opt_vol:.2%}")



import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Efficient Frontier

num_assets = len(tickers)

# Choose target returns for the frontier ----------

min_ret = annual_returns.min()
max_ret = annual_returns.max()

# Creating, say, 30 target returns between min_ret and max_ret
target_returns = np.linspace(min_ret, max_ret, 30)

# Lists to store the results for plotting
frontier_vols = []     # x-axis (risk)
frontier_rets = []     # y-axis (return)
frontier_weights = []  # store weights as well (optional, for inspection)

# constraint that weights sum to 1 
def weights_sum_to_one(weights):
    return np.sum(weights) - 1

# Bounds: no shorting
bounds = [(0, 1)] * num_assets

# Starting guess: equal weights
x0 = np.array([1 / num_assets] * num_assets)

# ---------- 3) Loop over target returns and solve optimization ----------
for target in target_returns:
    # For each target return, we set up TWO constraints:
    #   1) weights sum to 1
    #   2) portfolio_return(weights) = target
    constraints = (
        {"type": "eq", "fun": weights_sum_to_one},
        {
            "type": "eq",
            "fun": lambda w, mu=target: portfolio_return(w, annual_returns) - mu
        }
    )

    result = minimize(
        fun=portfolio_vol,           # minimize volatility
        x0=x0,                       # initial guess
        args=(annual_cov_matrix,),   # cov matrix for portfolio_vol
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if result.success:
        w_opt = result.x
        sigma = portfolio_vol(w_opt, annual_cov_matrix)
        mu = portfolio_return(w_opt, annual_returns)

        frontier_vols.append(sigma)
        frontier_rets.append(mu)
        frontier_weights.append(w_opt)
    else:
        # If optimization fails for some target, skip it
        print("⚠️ Frontier optimization failed for target:", target)


# Plotting EF

plt.figure(figsize=(10, 6))

# Plot the efficient frontier as a line of points
plt.plot(frontier_vols, frontier_rets, "o-", label="Efficient Frontier")

# Optional: equal-weight portfolio
equal_weights = np.array([1 / num_assets] * num_assets)
eq_ret = portfolio_return(equal_weights, annual_returns)
eq_vol = portfolio_vol(equal_weights, annual_cov_matrix)
plt.scatter(eq_vol, eq_ret, color="green", marker="s", s=80, label="Equal-Weight")

# Optional: min-vol portfolio (if you've already computed opt_weights)
try:
    minvol_ret = portfolio_return(opt_weights, annual_returns)
    minvol_vol = portfolio_vol(opt_weights, annual_cov_matrix)
    plt.scatter(minvol_vol, minvol_ret, color="red", marker="*", s=200, label="Min-Vol Portfolio")
except NameError:
    pass  # if opt_weights not defined, just skip

plt.title("Efficient Frontier (No Shorting)")
plt.xlabel("Annualized Volatility (Risk)")
plt.ylabel("Annualized Return")
plt.legend()
plt.grid(True)
plt.show()
