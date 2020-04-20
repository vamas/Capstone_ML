import pandas as pd
from sklearn.grid_search import GridSearchCV
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler, RobustScaler



def select_random_date(date_range, training_period, testing_period):
    cond = True
    while cond:
        #select random date
        training_start_date = np.random.choice(date_range)
        #check if there is enough days after 
        if (date_range.index(training_start_date) < (len(date_range) - training_period - testing_period)):
            cond = False
    return training_start_date    

def annualised_sharpe(returns, N=252):
    """
    Calculate the annualised Sharpe ratio of a returns stream 
    based on a number of trading periods, N. N defaults to 252,
    which then assumes a stream of daily returns.

    The function assumes that the returns are the excess of 
    those compared to a benchmark.
    """
    return np.sqrt(N) * returns.mean() / returns.std()


def equity_sharpe(ticker, date):
    """
    Calculates the annualised Sharpe ratio based on the daily
    returns of an equity ticker symbol listed in Yahoo Finance.

    The dates have been hardcoded here for the QuantStart article 
    on Sharpe ratios.
    """

    # Obtain the equities daily historic data for the desired time period
    # and add to a pandas DataFrame
    full_df = stockData.raw_financial_data[ticker]
    pdf = full_df[full_df.index <= date]

    # Use the percentage change method to easily calculate daily returns
    pdf['daily_ret'] = pdf['Adj Close'].pct_change()

    # Assume an average annual risk-free rate over the period of 5%
    pdf['excess_daily_ret'] = pdf['daily_ret'] - 0.05/252

    # Return the annualised Sharpe ratio based on the excess daily returns
    return annualised_sharpe(pdf['excess_daily_ret'])
