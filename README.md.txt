# Portfolio Optimisation & Efficient Frontier (Python)

Small project to explore basic portfolio construction and risk/return analysis.

## What it does

- Downloads historical prices for a list of tickers using `yfinance`
- Computes daily and annualised returns and volatilities
- Builds covariance and correlation matrices
- Calculates portfolio return and risk for a given set of weights
- Uses constrained optimisation (SciPy, SLSQP) to:
  - find the minimum-volatility portfolio
  - trace out an efficient frontier (no shorting)
- Plots the efficient frontier and highlights:
  - equal-weight portfolio
  - minimum-volatility portfolio

## Files

- `portfolio_optimizer.py` – main script with all logic
- `requirements.txt` – Python dependencies
- `.gitignore` – ignore compiled files, environments and data

## How to run

```bash
pip install -r requirements.txt
python portfolio_optimizer.py
