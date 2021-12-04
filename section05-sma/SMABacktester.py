import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute
plt.style.use("seaborn")

class SMABacktester():
    ''' Class for the vectorised backtesting of SMA-based trading strategies

    Attributes
    ==========
    symbol: str
        ticker symbol with which to work with
    SMA_S: int
        time window in days for shorter SMA
    SMA_L: int
        time window in days for longer SMA
    start: str
        start date for data retrieval
    end: str
        end date for data retrieval

    
    Methods
    =======
    get_data:
        retrieves and prepares the data
    
    set_parameters:
        sets one or two new SMA parameters

    test_strategy:
        runs the backtest for the SMA-based strategy

    plot_results:
        plots the performance of the strategy compared to buy and hold

    update_and_run:
        updates SMA parameters and returns the negative absolute performance (for minimiation algorithm)

    optimize_parameters:
        implements a brute force optimisation for the two SMA parameters
    '''

    def __init__(self, symbol, SMA_S, SMA_L, start, end):
        self.symbol = symbol
        self.SMA_S = SMA_S
        self.SMA_L = SMA_L
        self.start = start
        self.end = end
        self.results = None 
        self.get_data()


    def get_data(self):
        '''
        Retrieves and prepares the data
        '''
        raw = pd.read_csv("forex_pairs.csv", parse_dates=["Date"], index_col="Date")
        raw = raw[[self.symbol]].dropna()
        raw = raw.loc[self.start:self.end]
        raw.rename(columns={self.symbol:"price"}, inplace=True)
        raw["returns"] = np.log(raw / raw.shift(1))
        raw["SMA_S"] = raw["price"].rolling(self.SMA_S).mean()
        raw["SMA_L"] = raw["price"].rolling(self.SMA_L).mean()
        self.data = raw

    def set_parameters(self, SMA_S=None, SMA_L=None):
        '''
        Update SMA parameters and resp. time series
        '''
        if SMA_S is not None:
            self.SMA_S = SMA_S
            self.data["SMA_S"] = self.data["price"].rolling(self.SMA_S).mean()
        if SMA_L is not None:
            self.SMA_L= SMA_L
            self.data["SMA_L"] = self.data["price"].rolling(self.SMA_L).mean()

    def test_strategy(self):
        '''Backtests the trading strategy
        '''
        data = self.data.copy().dropna()
        data["position"]=np.where(data["SMA_S"] > data["SMA_L"], 1, -1)
        data["strategy"] = data["position"].shift(1)* data["returns"]
        data.dropna(inplace=True)
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data 
        # absolute performance of the strategy
        perf = data["cstrategy"].iloc[-1]
        # out-/underpeformance of the strategy
        outperf = perf - data["creturns"].iloc[-1]
        return round(perf, 6), round(outperf, 6)

