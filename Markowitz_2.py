"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=20, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def sma(self, index, asset, days=20):
        if index < days:
            return 0
        else:
            return self.price[asset].iloc[index-days:index].mean()
    
    def momentum(self, index, asset, days=20):
        if index < days:
            return 0
        else:
            return (self.price[asset].iloc[index-1] - self.price[asset].iloc[index-days]) / self.price[asset].iloc[index-days]
    
    def rsi(self, index, asset, days=14):
        if index < days:
            return 0
        gain = 0
        loss = 0
        for i in range(index - days, index):
            change = self.returns[asset].iloc[i]
            if change > 0:
                gain += change
            else:
                loss -= change
        if loss == 0:
            return 100
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def volatility(self, index, asset, days=20):
        if index < days:
            return 0
        return self.returns[asset].iloc[index-days:index].std()

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """

        x = []
        y = []
        n_samples = 1024
        for i in range(n_samples):
            idx = np.random.randint(self.lookback, len(self.price)-5)
            asset = np.random.choice(assets)
            x.append([
                self.price[asset].iloc[idx-1],
                self.sma(idx, asset),
                self.momentum(idx, asset),
                self.rsi(idx, asset),
                self.volatility(idx, asset)
            ])
            y.append(self.returns[asset].iloc[idx:idx+5].sum())

        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=20, random_state=42)
        model.fit(x, y)
        for i in range(self.lookback, len(self.price)):
            score = {}
            for asset in assets:
                features = [
                    self.price[asset].iloc[i-1],
                    self.sma(i, asset),
                    self.momentum(i, asset),
                    self.rsi(i, asset),
                    self.volatility(i, asset)
                ]
                score[asset] = model.predict([features])[0]
                score[asset] = max(score[asset], 0)
                
            for asset in assets:
                if sum(score.values()) == 0:
                    self.portfolio_weights[asset].iloc[i] = 1 / len(assets)
                else:
                    self.portfolio_weights[asset].iloc[i] = score[asset] / sum(score.values())

        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
