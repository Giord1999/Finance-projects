import pandas as pd 
import yfinance as yf 
import pandas_datareader as pdr
import matplotlib.pyplot as plt 
import datetime as dt 
import mplfinance as mpf
import numpy as np
import talib as ta
import scipy.optimize as opt

class StockAnalyzer:
    def __init__(self, ticker, period, interval, initial_capital):
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.initial_capital = initial_capital
        self.data = None
        self.combined_df = None
        self.backtest_results = None
        self.fib_levels = None

    def fetch_data(self):
        stock = yf.Ticker(self.ticker)
        self.data = stock.history(period=self.period, interval=self.interval)
        if self.data.empty:
            raise ValueError("No data available for the given ticker and period.")
        self.combined_df = pd.DataFrame(index=self.data.index)
        self.combined_df['Close'] = self.data["Close"]

    def calculate_indicators(self):
        # RSI
        self.combined_df["RSI"] = ta.RSI(self.data["Close"], timeperiod=14)
        
        # MACD
        macd, signal, histogram = ta.MACD(self.data["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
        self.combined_df["MACD"] = macd
        self.combined_df["Signal Line"] = signal
        self.combined_df["Histogram"] = histogram
        
        # Bollinger Bands
        upper, middle, lower = ta.BBANDS(self.data["Close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        self.combined_df["MA20"] = middle
        self.combined_df["Upper Band"] = upper
        self.combined_df["Lower Band"] = lower
        
        # Stochastic Oscillator
        slowk, slowd = ta.STOCH(self.data["High"], self.data["Low"], self.data["Close"], fastk_period=14, slowk_period=3, slowd_period=3)
        self.combined_df["%K"] = slowk
        self.combined_df["%D"] = slowd
        
        # Williams %R
        self.combined_df["Williams %R"] = ta.WILLR(self.data["High"], self.data["Low"], self.data["Close"], timeperiod=14)
        
        # CCI
        self.combined_df["CCI"] = ta.CCI(self.data["High"], self.data["Low"], self.data["Close"], timeperiod=20)
        
        # ADX
        self.combined_df["ADX"] = ta.ADX(self.data["High"], self.data["Low"], self.data["Close"], timeperiod=14)
        
        # OBV
        self.combined_df["OBV"] = ta.OBV(self.data["Close"], self.data["Volume"])
        
        # Ichimoku Cloud
        high9 = self.data["High"].rolling(window=9).max()
        low9 = self.data["Low"].rolling(window=9).min()
        self.combined_df["Tenkan-sen"] = (high9 + low9) / 2
        
        high26 = self.data["High"].rolling(window=26).max()
        low26 = self.data["Low"].rolling(window=26).min()
        self.combined_df["Kijun-sen"] = (high26 + low26) / 2
        
        self.combined_df["Senkou Span A"] = ((self.combined_df["Tenkan-sen"] + self.combined_df["Kijun-sen"]) / 2).shift(26)
        
        high52 = self.data["High"].rolling(window=52).max()
        low52 = self.data["Low"].rolling(window=52).min()
        self.combined_df["Senkou Span B"] = ((high52 + low52) / 2).shift(26)
        
        self.combined_df["Chikou Span"] = self.data["Close"].shift(-26)
        
        # Parabolic SAR
        self.combined_df["Parabolic SAR"] = ta.SAR(self.data["High"], self.data["Low"], acceleration=0.02, maximum=0.2)
        
        # Fibonacci Retracements
        fib_high = self.data["High"].max()
        fib_low = self.data["Low"].min()
        fib_range = fib_high - fib_low
        self.fib_levels = {
            "0%": fib_high,
            "23.6%": fib_high - 0.236 * fib_range,
            "38.2%": fib_high - 0.382 * fib_range,
            "50%": fib_high - 0.5 * fib_range,
            "61.8%": fib_high - 0.618 * fib_range,
            "100%": fib_low
        }

    def backtest_strategy(self):
        capital = self.initial_capital
        position = 0
        entry_price = 0
        trades = []
        cumulative_profit = [0]
        
        for i in range(1, len(self.combined_df)):
            rsi = self.combined_df["RSI"].iloc[i]
            price = self.combined_df["Close"].iloc[i]
            
            if rsi < 30 and position == 0:
                position = 1
                entry_price = price
                trades.append({"type": "buy", "price": price, "date": self.combined_df.index[i]})
            
            elif rsi > 70 and position == 1:
                profit = (price - entry_price) / entry_price * capital
                capital += profit
                position = 0
                trades.append({"type": "sell", "price": price, "date": self.combined_df.index[i], "profit": profit})
                cumulative_profit.append(cumulative_profit[-1] + profit)
            else:
                cumulative_profit.append(cumulative_profit[-1])
        
        total_return = (capital - self.initial_capital) / self.initial_capital * 100
        num_trades = len(trades) // 2
        max_drawdown = 0
        peak = cumulative_profit[0]
        for profit in cumulative_profit:
            if profit > peak:
                peak = profit
            drawdown = peak - profit
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        self.backtest_results = {
            "final_capital": capital,
            "total_return": total_return,
            "num_trades": num_trades,
            "max_drawdown": max_drawdown,
            "cumulative_profit": cumulative_profit,
            "trades": trades
        }


    def optimize_strategy(self, rsi_buy_range=(20, 40), rsi_sell_range=(60, 80)):
        def objective(params):
            rsi_buy, rsi_sell = params
            # Simula backtest con nuovi parametri
            capital = self.initial_capital
            position = 0
            entry_price = 0
            for i in range(1, len(self.combined_df)):
                rsi = self.combined_df["RSI"].iloc[i]
                price = self.combined_df["Close"].iloc[i]
                if rsi < rsi_buy and position == 0:
                    position = 1
                    entry_price = price
                elif rsi > rsi_sell and position == 1:
                    profit = (price - entry_price) / entry_price * capital
                    capital += profit
                    position = 0
            return -(capital - self.initial_capital)  # Negativo per massimizzare
        
        bounds = [(rsi_buy_range[0], rsi_buy_range[1]), (rsi_sell_range[0], rsi_sell_range[1])]
        result = opt.minimize(objective, [30, 70], bounds=bounds)
        optimal_buy, optimal_sell = result.x
        print(f"Optimal RSI Buy: {optimal_buy:.2f}, Sell: {optimal_sell:.2f}")
        return optimal_buy, optimal_sell

    def calculate_risk_metrics(self):
        if self.backtest_results is None:
            raise ValueError("Run backtest first.")
        
        # Calcola rendimenti giornalieri
        daily_returns = self.combined_df["Close"].pct_change().dropna()
        mean_return = daily_returns.mean()
        std_return = daily_returns.std()
        
        # Sharpe Ratio (assumendo tasso risk-free = 0)
        sharpe_ratio = mean_return / std_return * np.sqrt(252)  # Annualizzato
        
        # Sortino Ratio (solo downside volatility)
        downside_returns = daily_returns[daily_returns < 0]
        sortino_ratio = mean_return / downside_returns.std() * np.sqrt(252)
        
        # Volatilità annualizzata
        volatility = std_return * np.sqrt(252)
        
        self.backtest_results.update({
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "volatility": volatility
        })

        
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {sortino_ratio:.2f}")
        print(f"Volatility: {volatility:.2f}")
        

    def compare_with_benchmark(self, benchmark_ticker="^GSPC"):
        benchmark = yf.Ticker(benchmark_ticker).history(period=self.period, interval=self.interval)["Close"]
        benchmark_returns = benchmark.pct_change().dropna()
        stock_returns = self.combined_df["Close"].pct_change().dropna()
        
        # Calcola beta e alpha (semplificato)
        covariance = np.cov(stock_returns, benchmark_returns)[0][1]
        variance = np.var(benchmark_returns)
        beta = covariance / variance
        alpha = stock_returns.mean() - beta * benchmark_returns.mean()
        
        print(f"Beta: {beta:.2f}")
        print(f"Alpha: {alpha:.4f}")

    def print_results(self):
        print(f"Backtesting Results for {self.ticker}:")
        print(f"Initial Capital: ${self.initial_capital}")
        print(f"Final Capital: ${self.backtest_results['final_capital']:.2f}")
        print(f"Total Return: {self.backtest_results['total_return']:.2f}%")
        print(f"Number of Trades: {self.backtest_results['num_trades']}")
        print(f"Max Drawdown: ${self.backtest_results['max_drawdown']:.2f}")

    def plot_charts(self):
        # Plot principale con sottografici
        plt.figure(figsize=(12,18))
        ax1 = plt.subplot(611)
        ax1.set_title("Stock Price with Bollinger Bands for {}".format(self.ticker), color="white")
        ax1.plot(self.combined_df.index, self.combined_df["Close"], color="lightgray", label="Close")
        ax1.plot(self.combined_df.index, self.combined_df["MA20"], color="blue", label="MA20")
        ax1.plot(self.combined_df.index, self.combined_df["Upper Band"], color="red", label="Upper Band")
        ax1.plot(self.combined_df.index, self.combined_df["Lower Band"], color="green", label="Lower Band")
        ax1.legend()
        ax1.grid(True, color="#555555")
        ax1.set_axisbelow(True)
        ax1.set_facecolor("black")
        ax1.figure.set_facecolor("#121212")
        ax1.tick_params(axis="x", colors="white")
        ax1.tick_params(axis="y", colors="white")

        ax2 = plt.subplot(612, sharex=ax1)
        ax2.set_title("RSI value for {}".format(self.ticker), color="white")
        ax2.plot(self.combined_df.index, self.combined_df["RSI"], color="lightgray")
        ax2.axhline(0, linestyle="--", alpha=0.5, color="#ff0000")
        ax2.axhline(10, linestyle="--", alpha=0.5, color="#ffaa00")
        ax2.axhline(20, linestyle="--", alpha=0.5, color="#00ff00")
        ax2.axhline(30, linestyle="--", alpha=0.5, color="#cccccc")
        ax2.axhline(70, linestyle="--", alpha=0.5, color="#cccccc")
        ax2.axhline(80, linestyle="--", alpha=0.5, color="#00ff00")
        ax2.axhline(90, linestyle="--", alpha=0.5, color="#ffaa00")
        ax2.axhline(100, linestyle="--", alpha=0.5, color="#ff0000")
        ax2.grid(False)
        ax2.set_axisbelow(True)
        ax2.set_facecolor("black")
        ax2.tick_params(axis="x", colors="white")
        ax2.tick_params(axis="y", colors="white")

        ax3 = plt.subplot(613, sharex=ax1)
        ax3.set_title("MACD for {}".format(self.ticker), color="white")
        ax3.plot(self.combined_df.index, self.combined_df["MACD"], color="blue", label="MACD")
        ax3.plot(self.combined_df.index, self.combined_df["Signal Line"], color="red", label="Signal Line")
        ax3.bar(self.combined_df.index, self.combined_df["Histogram"], color="gray", alpha=0.5, label="Histogram")
        ax3.legend()
        ax3.grid(True, color="#555555")
        ax3.set_axisbelow(True)
        ax3.set_facecolor("black")
        ax3.tick_params(axis="x", colors="white")
        ax3.tick_params(axis="y", colors="white")

        ax4 = plt.subplot(614, sharex=ax1)
        ax4.set_title("Stochastic Oscillator for {}".format(self.ticker), color="white")
        ax4.plot(self.combined_df.index, self.combined_df["%K"], color="blue", label="%K")
        ax4.plot(self.combined_df.index, self.combined_df["%D"], color="red", label="%D")
        ax4.axhline(20, linestyle="--", alpha=0.5, color="#cccccc")
        ax4.axhline(80, linestyle="--", alpha=0.5, color="#cccccc")
        ax4.legend()
        ax4.grid(True, color="#555555")
        ax4.set_axisbelow(True)
        ax4.set_facecolor("black")
        ax4.tick_params(axis="x", colors="white")
        ax4.tick_params(axis="y", colors="white")

        ax5 = plt.subplot(615, sharex=ax1)
        ax5.set_title("Williams %R and CCI for {}".format(self.ticker), color="white")
        ax5.plot(self.combined_df.index, self.combined_df["Williams %R"], color="green", label="Williams %R")
        ax5.plot(self.combined_df.index, self.combined_df["CCI"], color="orange", label="CCI")
        ax5.axhline(-20, linestyle="--", alpha=0.5, color="#cccccc")
        ax5.axhline(-80, linestyle="--", alpha=0.5, color="#cccccc")
        ax5.axhline(100, linestyle="--", alpha=0.5, color="#cccccc")
        ax5.axhline(-100, linestyle="--", alpha=0.5, color="#cccccc")
        ax5.legend()
        ax5.grid(True, color="#555555")
        ax5.set_axisbelow(True)
        ax5.set_facecolor("black")
        ax5.tick_params(axis="x", colors="white")
        ax5.tick_params(axis="y", colors="white")

        ax6 = plt.subplot(616, sharex=ax1)
        ax6.set_title("ADX and OBV for {}".format(self.ticker), color="white")
        ax6.plot(self.combined_df.index, self.combined_df["ADX"], color="purple", label="ADX")
        ax6.axhline(25, linestyle="--", alpha=0.5, color="#cccccc")
        ax6.legend()
        ax6.grid(True, color="#555555")
        ax6.set_axisbelow(True)
        ax6.set_facecolor("black")
        ax6.tick_params(axis="x", colors="white")
        ax6.tick_params(axis="y", colors="white")

        ax6_twin = ax6.twinx()
        ax6_twin.plot(self.combined_df.index, self.combined_df["OBV"], color="cyan", label="OBV")
        ax6_twin.tick_params(axis="y", colors="white")
        ax6_twin.legend(loc="upper right")

        plt.tight_layout()
        plt.show()

        # Candlestick plot
        mpf.plot(data=self.data, type="candle", style="yahoo", volume=True, title="Candlestick chart for {}".format(self.ticker), mav=(20,50), figratio=(12,8), figscale=1.5)

        # Plot Ichimoku, Parabolic SAR, Fibonacci
        plt.figure(figsize=(12, 8))
        ax_new = plt.subplot(111)
        ax_new.set_title("Price with Ichimoku Cloud, Parabolic SAR, and Fibonacci Retracements for {}".format(self.ticker), color="white")
        ax_new.plot(self.combined_df.index, self.combined_df["Close"], color="lightgray", label="Close")
        ax_new.plot(self.combined_df.index, self.combined_df["Tenkan-sen"], color="blue", label="Tenkan-sen")
        ax_new.plot(self.combined_df.index, self.combined_df["Kijun-sen"], color="red", label="Kijun-sen")
        ax_new.fill_between(self.combined_df.index, self.combined_df["Senkou Span A"], self.combined_df["Senkou Span B"], where=self.combined_df["Senkou Span A"] >= self.combined_df["Senkou Span B"], color="green", alpha=0.3, label="Cloud (Bullish)")
        ax_new.fill_between(self.combined_df.index, self.combined_df["Senkou Span A"], self.combined_df["Senkou Span B"], where=self.combined_df["Senkou Span A"] < self.combined_df["Senkou Span B"], color="red", alpha=0.3, label="Cloud (Bearish)")
        ax_new.plot(self.combined_df.index, self.combined_df["Chikou Span"], color="purple", label="Chikou Span")
        ax_new.scatter(self.combined_df.index, self.combined_df["Parabolic SAR"], color="orange", s=1, label="Parabolic SAR")
        for level, value in self.fib_levels.items():
            ax_new.axhline(value, linestyle="--", alpha=0.7, label=f"Fib {level}", color="cyan")
        ax_new.legend()
        ax_new.grid(True, color="#555555")
        ax_new.set_axisbelow(True)
        ax_new.set_facecolor("black")
        ax_new.figure.set_facecolor("#121212")
        ax_new.tick_params(axis="x", colors="white")
        ax_new.tick_params(axis="y", colors="white")
        plt.tight_layout()
        plt.show()

        # Cumulative profit plot
        plt.figure(figsize=(12, 6))
        x = self.combined_df.index[1:]
        y = self.backtest_results["cumulative_profit"][1:]
        if len(x) != len(y):
            min_len = min(len(x), len(y))
            x = x[:min_len]
            y = y[:min_len]
        plt.plot(x, y, color="blue", label="Cumulative Profit")
        plt.title(f"Cumulative Profit from Backtesting for {self.ticker}", color="white")
        plt.xlabel("Date")
        plt.ylabel("Profit ($)")
        plt.grid(True, color="#555555")
        plt.gca().set_facecolor("black")
        plt.gcf().set_facecolor("#121212")
        plt.tick_params(axis="x", colors="white")
        plt.tick_params(axis="y", colors="white")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def run(self):
        self.fetch_data()
        self.calculate_indicators()
        self.backtest_strategy()
        self.print_results()
        self.plot_charts()
        
        # Aggiungi chiamate alle nuove funzionalità
        print("\n--- Ottimizzazione Strategia ---")
        optimal_buy, optimal_sell = self.optimize_strategy()
        
        print("\n--- Metriche di Rischio ---")
        self.calculate_risk_metrics()
        
        print("\n--- Confronto con Benchmark ---")
        self.compare_with_benchmark()


# Esempio di utilizzo (sostituisci con input reali)
if __name__ == "__main__":
    ticker = input("Enter the ticker symbol: ")
    period = input("Enter the period (e.g., '1y' for 1 year, '6mo' for 6 months): ")
    interval = input("Enter the data interval (e.g., '1d' for daily, '1wk' for weekly): ")
    initial_capital = int(input("Enter the initial capital for backtesting (e.g., 10000): "))
    analyzer = StockAnalyzer(ticker, period, interval, initial_capital)
    analyzer.run()
