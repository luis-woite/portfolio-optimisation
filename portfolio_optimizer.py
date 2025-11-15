# ============================================================
# 🧮 Portfolio Risk & Return Analyzer – Step 1
# ------------------------------------------------------------
# GOAL:
#   1. Download daily prices for several stocks
#   2. Compute daily % returns
#   3. Calculate annualized return & volatility
#   4. Build covariance and correlation matrices
# ============================================================

# ---------- 1) Import required libraries ----------
# pandas: handles data tables (like Excel, but better)
# numpy:  numerical calculations (e.g., square roots)
# yfinance:  download stock price data from Yahoo Finance

import yfinance as yf
import pandas as pd
import numpy as np

from scipy.optimize import minimize


# ---------- 2) Choose which stocks we want ----------
# These tickers are just examples — all large U.S. companies.
# You can replace them later (e.g. ["TSLA","NVDA","AMZN","MSFT","JPM"])
tickers = ["AAPL", "MSFT", "META", "GOOG", "XOM"]

# ---------- 3) Download daily closing prices ----------
# 'start' and 'end' define our time window.
# progress=False hides the progress bar to keep output clean.
data = yf.download(tickers,
                   start="2024-01-01",
                   end="2024-12-31",
                   progress=False)["Close"]

# ---------- 4) Clean the data ----------
# Sometimes some days are missing for certain tickers.
# dropna() removes any rows with missing prices.
data = data.dropna()

# ---------- 5) Take a quick look ----------
print("✅ First few rows of daily closing prices:")
print(data.head())

# ============================================================
# 🧾 STEP 2: Compute daily percentage returns
# ------------------------------------------------------------
# pct_change() gives (P_t / P_{t-1} - 1)
# e.g., if price moves from 100 → 101, return = 0.01 = 1%
# We drop the first row since it will be NaN (no previous day).
# ============================================================

daily_returns = data.pct_change().dropna()

print("\n✅ First few rows of daily returns (in decimals):")
print(daily_returns.head())

# ============================================================
# 📈 STEP 3: Annualize mean returns and volatility
# ------------------------------------------------------------
# Convention: there are roughly 252 trading days per year.

"""
if returns are i.i.d., the variance of the sum of 252 returns is: Var(R_annaual) = 252 * Var(R_daily)
(bc variances add up for independent variables)

vola is sd of variance, so: sd(R_annual) = sqrt(252 * Var(R_daily)) = sqrt(252) * sd(R_daily)

intuition:
- Mean return scales linearly with time (you earn roughly 252 times the daily return per year on average).

- Volatility scales with the square root of time, because uncertainty accumulates like a random walk — the “range” of possible outcomes spreads out slower (gradually) than linearly.


"""

# Annualized Return ≈ mean_daily_return × 252
# Annualized Volatility ≈ std_daily_return × √252
# ============================================================

trading_days = 252

# mean() computes the average daily return for each stock
mean_daily = daily_returns.mean()

# Convert daily metrics to annual metrics
annual_returns = mean_daily * trading_days
annual_vol = daily_returns.std() * np.sqrt(trading_days)

print("\n✅ Annualized Expected Returns (per stock):")
print(annual_returns)

print("\n✅ Annualized Volatilities (per stock):")
print(annual_vol)

# ============================================================
# 🧠 STEP 4: Covariance and Correlation matrices
# ------------------------------------------------------------
# Covariance: measures how two assets move together (raw scale)
# Correlation: standardized version between -1 and +1
#   +1 = move exactly together
#   0  = no relationship
#   -1 = move opposite
# ------------------------------------------------------------
# Lower correlation → better diversification benefits
# ============================================================

cov_matrix = daily_returns.cov()
corr_matrix = daily_returns.corr()



print("\n✅ Covariance matrix (daily):")
print(cov_matrix)

print("\n✅ Correlation matrix:")
print(corr_matrix)

# ============================================================
# ✅ Summary:
# You now have:
#   - prices (data)
#   - daily returns (daily_returns)
#   - annualized returns and volatilities (annual_returns, annual_vol)
#   - covariance and correlation matrices (cov_matrix, corr_matrix)
# These are the building blocks for portfolio optimization.
# ============================================================




# ============================================================
# 🧮 STEP 5: Portfolio return and risk for a given set of weights
# ------------------------------------------------------------
# IDEA:
#   - each stock has an annual return (we computed that)
#   - we choose how much money to put in each stock (weights)
#   - portfolio return = weighted average of individual returns
#   - portfolio risk   = sqrt( w' * Σ * w )
#     where:
#        w  = weights column vector
#        Σ  = covariance matrix (we have daily, so we annualize it)
# ============================================================

# ---------- 1) choose some weights ----------
# We have 5 tickers, so we need 5 weights.
# Here we just split money equally: 20% in each stock.
# NOTE: weights must sum to 1.
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

# quick safety check
if not np.isclose(weights.sum(), 1.0):
    raise ValueError("Weights must sum to 1.0!")

# ---------- 2) portfolio annual return ----------
# annual_returns is a pandas Series with index = tickers.
# We align weights with annual_returns by position.
portfolio_return = np.dot(weights, annual_returns)

# ---------- 3) portfolio annual volatility ----------
# cov_matrix is currently DAILY covariance.
# To get ANNUAL covariance, multiply by number of trading days.
annual_cov_matrix = cov_matrix * trading_days

# Formula: σ_p = sqrt( w' Σ w )
portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(annual_cov_matrix, weights)))

# ---------- 4) print results ----------
print("\n===== Portfolio (equal weights) =====")
print(f"Weights: {dict(zip(tickers, weights))}")
print(f"Expected annual return: {portfolio_return:.2%}")
print(f"Expected annual volatility: {portfolio_vol:.2%}")



# ============================================================
# ⚙️ Helper functions for portfolio math
# ============================================================

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

# ============================================================
# 🧠 STEP 6: Find the minimum-volatility portfolio
# ------------------------------------------------------------
# We tell scipy:
#   "Here are 5 assets. Find weights w1..w5 that minimize portfolio_vol
#    subject to:
#       - sum(weights) = 1
#       - 0 <= weight <= 1  (no shorting)
# ============================================================

num_assets = len(tickers)

# starting guess: equal weights
x0 = np.array([1/num_assets] * num_assets)

# constraint 1: weights must sum to 1
constraints = ({
    "type": "eq",               # equality constraint
    "fun": lambda w: np.sum(w) - 1
})

# bounds: each weight between 0 and 1
# (you can change this later to allow shorting)
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

# check if it worked
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

# ============================================================
# 🧮 STEP 7: Efficient Frontier
# ============================================================

num_assets = len(tickers)

# ---------- 1) Choose target returns for the frontier ----------
# We'll span from the minimum to the maximum individual asset return.
# You can tighten or widen this range later if you like.
min_ret = annual_returns.min()
max_ret = annual_returns.max()

# Let's create, say, 30 target returns between min_ret and max_ret
target_returns = np.linspace(min_ret, max_ret, 30)

# Lists to store the results for plotting
frontier_vols = []     # x-axis (risk)
frontier_rets = []     # y-axis (return)
frontier_weights = []  # store weights as well (optional, for inspection)

# ---------- 2) Helper: constraint that weights sum to 1 ----------
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


# ============================================================
# 🎨 Plot the Efficient Frontier
# ============================================================

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
