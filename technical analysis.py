import pandas as pd 
import yfinance as yf 
import pandas_datareader as pdr
import matplotlib.pyplot as plt 
import datetime as dt 
import mplfinance as mpf
import numpy as np
import time
import talib as ta
import scipy.optimize as opt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

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

    def backtest_strategy(self, commission_rate=0.001):
        """Esegue il backtesting con strategia migliorata e metriche aggiuntive."""
        capital = self.initial_capital
        position = 0
        entry_price = 0
        trades = []
        cumulative_profit = [0]
        wins = 0
        losses = 0
        total_win_amount = 0
        total_loss_amount = 0
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for i in range(1, len(self.combined_df)):
            rsi = self.combined_df["RSI"].iloc[i]
            macd = self.combined_df["MACD"].iloc[i]
            signal = self.combined_df["Signal Line"].iloc[i]
            price = self.combined_df["Close"].iloc[i]
            lower_band = self.combined_df["Lower Band"].iloc[i]
            
            # Strategia migliorata: RSI < 30 AND prezzo sotto Lower Bollinger AND MACD crossover (MACD > Signal)
            buy_condition = rsi < 30 and price < lower_band and macd > signal and position == 0
            sell_condition = rsi > 70 and position == 1
            
            if buy_condition:
                position = 1
                entry_price = price
                trades.append({"type": "buy", "price": price, "date": self.combined_df.index[i]})
            
            elif sell_condition:
                gross_profit = (price - entry_price) / entry_price * capital
                commission = commission_rate * capital  # Commission on capital
                net_profit = gross_profit - commission
                capital += net_profit
                position = 0
                trades.append({"type": "sell", "price": price, "date": self.combined_df.index[i], "profit": net_profit})
                cumulative_profit.append(cumulative_profit[-1] + net_profit)
                
                # Track wins/losses
                if net_profit > 0:
                    wins += 1
                    total_win_amount += net_profit
                    consecutive_wins += 1
                    consecutive_losses = 0
                    if consecutive_wins > max_consecutive_wins:
                        max_consecutive_wins = consecutive_wins
                else:
                    losses += 1
                    total_loss_amount += abs(net_profit)
                    consecutive_losses += 1
                    consecutive_wins = 0
                    if consecutive_losses > max_consecutive_losses:
                        max_consecutive_losses = consecutive_losses
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
        
        # Calcola metriche aggiuntive
        win_rate = (wins / num_trades * 100) if num_trades > 0 else 0
        profit_factor = (total_win_amount / total_loss_amount) if total_loss_amount > 0 else float('inf')
        avg_win = total_win_amount / wins if wins > 0 else 0
        avg_loss = total_loss_amount / losses if losses > 0 else 0
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else float('inf')
        
        self.backtest_results = {
            "final_capital": capital,
            "total_return": total_return,
            "num_trades": num_trades,
            "max_drawdown": max_drawdown,
            "cumulative_profit": cumulative_profit,
            "trades": trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses,
            "calmar_ratio": calmar_ratio
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
        time.sleep(1)
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
        print(f"Win Rate: {self.backtest_results['win_rate']:.2f}%")
        print(f"Profit Factor: {self.backtest_results['profit_factor']:.2f}")
        print(f"Average Win: ${self.backtest_results['avg_win']:.2f}")
        print(f"Average Loss: ${self.backtest_results['avg_loss']:.2f}")
        print(f"Max Consecutive Wins: {self.backtest_results['max_consecutive_wins']}")
        print(f"Max Consecutive Losses: {self.backtest_results['max_consecutive_losses']}")
        print(f"Calmar Ratio: {self.backtest_results['calmar_ratio']:.2f}")

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


class FundamentalAnalyzer:
    """
    Classe completa per l'analisi fondamentale dei ticker finanziari.
    Fornisce metriche chiave, valutazioni comparative e raccomandazioni di investimento.
    """
    
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        self.info = None
        self.financials = None
        self.balance_sheet = None
        self.cashflow = None
        self.quarterly_financials = None
        self.quarterly_balance_sheet = None
        self.quarterly_cashflow = None
        self.fundamental_metrics = {}
        self.valuation_metrics = {}
        self.quality_metrics = {}
        self.growth_metrics = {}
        self.risk_metrics = {}
        
    def fetch_fundamental_data(self):
        """Scarica tutti i dati fondamentali necessari per l'analisi."""
        try:
            print(f"Downloading fundamental data for {self.ticker}...")
            
            # Informazioni generali
            self.info = self.stock.info
            
            # Dati finanziari annuali
            self.financials = self.stock.financials
            self.balance_sheet = self.stock.balance_sheet
            self.cashflow = self.stock.cashflow
            
            # Dati finanziari trimestrali
            self.quarterly_financials = self.stock.quarterly_financials
            self.quarterly_balance_sheet = self.stock.quarterly_balance_sheet
            self.quarterly_cashflow = self.stock.quarterly_cashflow
            
            print("✓ Fundamental data downloaded successfully!")
            
        except Exception as e:
            print(f"❌ Error downloading fundamental data: {str(e)}")
            raise
    
    def calculate_valuation_metrics(self):
        """Calcola le principali metriche di valutazione."""
        try:
            current_price = self.info.get('currentPrice', self.info.get('regularMarketPrice', 0))
            shares_outstanding = self.info.get('sharesOutstanding', 0)
            market_cap = self.info.get('marketCap', current_price * shares_outstanding)
            
            # P/E Ratios
            pe_ratio = self.info.get('trailingPE', None)
            forward_pe = self.info.get('forwardPE', None)
            
            # Price-to-Book
            pb_ratio = self.info.get('priceToBook', None)
            
            # Price-to-Sales
            revenue_ttm = self.info.get('totalRevenue', 0)
            ps_ratio = market_cap / revenue_ttm if revenue_ttm > 0 else None
            
            # PEG Ratio
            peg_ratio = self.info.get('pegRatio', None)
            
            # Enterprise Value metrics
            enterprise_value = self.info.get('enterpriseValue', 0)
            ebitda = self.info.get('ebitda', 0)
            ev_ebitda = enterprise_value / ebitda if ebitda > 0 else None
            ev_revenue = enterprise_value / revenue_ttm if revenue_ttm > 0 else None
            
            self.valuation_metrics = {
                'current_price': current_price,
                'market_cap': market_cap,
                'pe_ratio': pe_ratio,
                'forward_pe': forward_pe,
                'pb_ratio': pb_ratio,
                'ps_ratio': ps_ratio,
                'peg_ratio': peg_ratio,
                'ev_ebitda': ev_ebitda,
                'ev_revenue': ev_revenue,
                'enterprise_value': enterprise_value
            }
            
        except Exception as e:
            print(f"❌ Error calculating valuation metrics: {str(e)}")
    
    def calculate_profitability_metrics(self):
        """Calcola le metriche di redditività."""
        try:
            # Margini di profitto
            gross_margin = self.info.get('grossMargins', 0) * 100 if self.info.get('grossMargins') else None
            operating_margin = self.info.get('operatingMargins', 0) * 100 if self.info.get('operatingMargins') else None
            profit_margin = self.info.get('profitMargins', 0) * 100 if self.info.get('profitMargins') else None
            
            # Return ratios
            roe = self.info.get('returnOnEquity', 0) * 100 if self.info.get('returnOnEquity') else None
            roa = self.info.get('returnOnAssets', 0) * 100 if self.info.get('returnOnAssets') else None
            roic = self.info.get('returnOnCapital', 0) * 100 if self.info.get('returnOnCapital') else None
            
            self.fundamental_metrics.update({
                'gross_margin': gross_margin,
                'operating_margin': operating_margin,
                'profit_margin': profit_margin,
                'roe': roe,
                'roa': roa,
                'roic': roic
            })
            
        except Exception as e:
            print(f"❌ Error calculating profitability metrics: {str(e)}")
    
    def calculate_financial_health_metrics(self):
        """Calcola le metriche di salute finanziaria."""
        try:
            # Debt ratios
            total_debt = self.info.get('totalDebt', 0)
            total_cash = self.info.get('totalCash', 0)
            net_debt = total_debt - total_cash
            
            market_cap = self.valuation_metrics.get('market_cap', 0)
            debt_to_equity = self.info.get('debtToEquity', None)
            
            # Liquidity ratios
            current_ratio = self.info.get('currentRatio', None)
            quick_ratio = self.info.get('quickRatio', None)
            
            # Interest coverage
            ebitda = self.info.get('ebitda', 0)
            interest_expense = abs(self.info.get('interestExpense', 0))
            interest_coverage = ebitda / interest_expense if interest_expense > 0 else None
            
            self.quality_metrics = {
                'total_debt': total_debt,
                'total_cash': total_cash,
                'net_debt': net_debt,
                'debt_to_equity': debt_to_equity,
                'current_ratio': current_ratio,
                'quick_ratio': quick_ratio,
                'interest_coverage': interest_coverage
            }
            
        except Exception as e:
            print(f"❌ Error calculating financial health metrics: {str(e)}")
    
    def calculate_growth_metrics(self):
        """Calcola le metriche di crescita."""
        try:
            # Revenue growth
            revenue_growth = self.info.get('revenueGrowth', 0) * 100 if self.info.get('revenueGrowth') else None
            
            # Earnings growth
            earnings_growth = self.info.get('earningsGrowth', 0) * 100 if self.info.get('earningsGrowth') else None
            
            # Quarterly growth
            revenue_growth_quarterly = self.info.get('revenueQuarterlyGrowth', 0) * 100 if self.info.get('revenueQuarterlyGrowth') else None
            earnings_growth_quarterly = self.info.get('earningsQuarterlyGrowth', 0) * 100 if self.info.get('earningsQuarterlyGrowth') else None
            
            self.growth_metrics = {
                'revenue_growth_yoy': revenue_growth,
                'earnings_growth_yoy': earnings_growth,
                'revenue_growth_qoq': revenue_growth_quarterly,
                'earnings_growth_qoq': earnings_growth_quarterly
            }
            
        except Exception as e:
            print(f"❌ Error calculating growth metrics: {str(e)}")
    
    def calculate_dividend_metrics(self):
        """Calcola le metriche sui dividendi."""
        try:
            dividend_yield = self.info.get('dividendYield', 0) * 100 if self.info.get('dividendYield') else 0
            dividend_rate = self.info.get('dividendRate', 0)
            payout_ratio = self.info.get('payoutRatio', 0) * 100 if self.info.get('payoutRatio') else None
            
            # Five year average dividend yield
            five_year_avg_yield = self.info.get('fiveYearAvgDividendYield', None)
            
            self.fundamental_metrics.update({
                'dividend_yield': dividend_yield,
                'dividend_rate': dividend_rate,
                'payout_ratio': payout_ratio,
                'five_year_avg_dividend_yield': five_year_avg_yield
            })
            
        except Exception as e:
            print(f"❌ Error calculating dividend metrics: {str(e)}")
    
    def calculate_efficiency_metrics(self):
        """Calcola le metriche di efficienza operativa."""
        try:
            # Asset turnover
            revenue_ttm = self.info.get('totalRevenue', 0)
            total_assets = self.info.get('totalAssets', 0)
            asset_turnover = revenue_ttm / total_assets if total_assets > 0 else None
            
            # Inventory turnover (se disponibile)
            inventory_turnover = self.info.get('inventoryTurnover', None)
            
            # Days sales outstanding
            receivables_turnover = self.info.get('receivablesTurnover', None)
            
            self.fundamental_metrics.update({
                'asset_turnover': asset_turnover,
                'inventory_turnover': inventory_turnover,
                'receivables_turnover': receivables_turnover
            })
            
        except Exception as e:
            print(f"❌ Error calculating efficiency metrics: {str(e)}")
    
    def analyze_competitive_position(self):
        """Analizza la posizione competitiva dell'azienda."""
        try:
            sector = self.info.get('sector', 'Unknown')
            industry = self.info.get('industry', 'Unknown')
            
            # Market position metrics
            market_cap = self.valuation_metrics.get('market_cap', 0)
            
            # Competitive advantages indicators
            brand_strength_indicators = {
                'gross_margin': self.fundamental_metrics.get('gross_margin', 0),
                'roe': self.fundamental_metrics.get('roe', 0),
                'roic': self.fundamental_metrics.get('roic', 0)
            }
            
            return {
                'sector': sector,
                'industry': industry,
                'market_cap_category': self._categorize_market_cap(market_cap),
                'competitive_strength_score': self._calculate_competitive_strength(brand_strength_indicators)
            }
            
        except Exception as e:
            print(f"❌ Error analyzing competitive position: {str(e)}")
            return {}
    
    def _categorize_market_cap(self, market_cap):
        """Categorizza l'azienda per dimensione di market cap."""
        if market_cap >= 200_000_000_000:
            return "Mega Cap (>$200B)"
        elif market_cap >= 10_000_000_000:
            return "Large Cap ($10B-$200B)"
        elif market_cap >= 2_000_000_000:
            return "Mid Cap ($2B-$10B)"
        elif market_cap >= 300_000_000:
            return "Small Cap ($300M-$2B)"
        else:
            return "Micro Cap (<$300M)"
    
    def _calculate_competitive_strength(self, indicators):
        """Calcola un punteggio di forza competitiva."""
        score = 0
        count = 0
        
        # Gross margin strength (>20% good, >40% excellent)
        if indicators.get('gross_margin'):
            if indicators['gross_margin'] > 40:
                score += 30
            elif indicators['gross_margin'] > 20:
                score += 20
            elif indicators['gross_margin'] > 10:
                score += 10
            count += 1
        
        # ROE strength (>15% good, >25% excellent)
        if indicators.get('roe'):
            if indicators['roe'] > 25:
                score += 35
            elif indicators['roe'] > 15:
                score += 25
            elif indicators['roe'] > 10:
                score += 15
            count += 1
        
        # ROIC strength (>12% good, >20% excellent)
        if indicators.get('roic'):
            if indicators['roic'] > 20:
                score += 35
            elif indicators['roic'] > 12:
                score += 25
            elif indicators['roic'] > 8:
                score += 15
            count += 1
        
        return score / max(count, 1) if count > 0 else 0
    
    def generate_investment_recommendation(self):
        """Genera una raccomandazione di investimento basata sull'analisi fondamentale."""
        try:
            recommendation_score = 0
            factors = []
            
            # Valuation attractiveness (30% weight)
            pe_ratio = self.valuation_metrics.get('pe_ratio')
            if pe_ratio:
                if pe_ratio < 15:
                    recommendation_score += 30
                    factors.append("✓ P/E attrattivo (<15)")
                elif pe_ratio < 25:
                    recommendation_score += 20
                    factors.append("○ P/E moderato (15-25)")
                else:
                    recommendation_score += 5
                    factors.append("⚠ P/E elevato (>25)")
            
            # Financial health (25% weight)
            current_ratio = self.quality_metrics.get('current_ratio')
            debt_to_equity = self.quality_metrics.get('debt_to_equity')
            
            if current_ratio and current_ratio > 1.5:
                recommendation_score += 15
                factors.append("✓ Liquidità solida")
            elif current_ratio and current_ratio > 1.0:
                recommendation_score += 10
                factors.append("○ Liquidità adeguata")
            
            if debt_to_equity is not None and debt_to_equity < 0.5:
                recommendation_score += 10
                factors.append("✓ Debito contenuto")
            elif debt_to_equity is not None and debt_to_equity < 1.0:
                recommendation_score += 5
                factors.append("○ Debito gestibile")
            
            # Profitability (25% weight)
            roe = self.fundamental_metrics.get('roe')
            profit_margin = self.fundamental_metrics.get('profit_margin')
            
            if roe and roe > 15:
                recommendation_score += 15
                factors.append("✓ ROE eccellente (>15%)")
            elif roe and roe > 10:
                recommendation_score += 10
                factors.append("○ ROE buono (10-15%)")
            
            if profit_margin and profit_margin > 10:
                recommendation_score += 10
                factors.append("✓ Margine di profitto solido")
            elif profit_margin and profit_margin > 5:
                recommendation_score += 5
                factors.append("○ Margine di profitto adeguato")
            
            # Growth prospects (20% weight)
            revenue_growth = self.growth_metrics.get('revenue_growth_yoy')
            earnings_growth = self.growth_metrics.get('earnings_growth_yoy')
            
            if revenue_growth and revenue_growth > 10:
                recommendation_score += 10
                factors.append("✓ Crescita ricavi solida")
            elif revenue_growth and revenue_growth > 0:
                recommendation_score += 5
                factors.append("○ Crescita ricavi positiva")
            
            if earnings_growth and earnings_growth > 15:
                recommendation_score += 10
                factors.append("✓ Crescita utili eccellente")
            elif earnings_growth and earnings_growth > 0:
                recommendation_score += 5
                factors.append("○ Crescita utili positiva")
            
            # Generate recommendation
            if recommendation_score >= 80:
                recommendation = "STRONG BUY"
                color = "🟢"
            elif recommendation_score >= 60:
                recommendation = "BUY"
                color = "🟢"
            elif recommendation_score >= 40:
                recommendation = "HOLD"
                color = "🟡"
            elif recommendation_score >= 20:
                recommendation = "WEAK HOLD"
                color = "🟠"
            else:
                recommendation = "SELL"
                color = "🔴"
            
            return {
                'recommendation': recommendation,
                'score': recommendation_score,
                'color': color,
                'factors': factors
            }
            
        except Exception as e:
            print(f"❌ Error generating recommendation: {str(e)}")
            return {'recommendation': 'INSUFFICIENT DATA', 'score': 0, 'factors': []}
    
    def print_comprehensive_analysis(self):
        """Stampa un'analisi completa dell'azienda."""
        print("=" * 80)
        print(f"🏢 ANALISI FONDAMENTALE COMPLETA - {self.ticker}")
        print("=" * 80)
        
        # Company overview
        print(f"\n📊 PANORAMICA AZIENDALE")
        print("-" * 40)
        company_name = self.info.get('longName', self.ticker)
        sector = self.info.get('sector', 'N/A')
        industry = self.info.get('industry', 'N/A')
        employees = self.info.get('fullTimeEmployees', 'N/A')
        
        print(f"Ragione Sociale: {company_name}")
        print(f"Settore: {sector}")
        print(f"Industria: {industry}")
        print(f"Dipendenti: {employees:,}" if isinstance(employees, int) else f"Dipendenti: {employees}")
        
        competitive_analysis = self.analyze_competitive_position()
        if competitive_analysis:
            print(f"Categoria Market Cap: {competitive_analysis.get('market_cap_category', 'N/A')}")
            print(f"Punteggio Competitività: {competitive_analysis.get('competitive_strength_score', 0):.1f}/100")
        
        # Valuation metrics
        print(f"\n💰 METRICHE DI VALUTAZIONE")
        print("-" * 40)
        current_price = self.valuation_metrics.get('current_price', 0)
        market_cap = self.valuation_metrics.get('market_cap', 0)
        
        print(f"Prezzo Attuale: ${current_price:.2f}")
        print(f"Market Cap: ${market_cap/1_000_000_000:.2f}B" if market_cap > 0 else "Market Cap: N/A")
        
        pe_ratio = self.valuation_metrics.get('pe_ratio')
        print(f"P/E Ratio: {pe_ratio:.2f}" if pe_ratio else "P/E Ratio: N/A")
        
        forward_pe = self.valuation_metrics.get('forward_pe')
        print(f"Forward P/E: {forward_pe:.2f}" if forward_pe else "Forward P/E: N/A")
        
        pb_ratio = self.valuation_metrics.get('pb_ratio')
        print(f"P/B Ratio: {pb_ratio:.2f}" if pb_ratio else "P/B Ratio: N/A")
        
        ev_ebitda = self.valuation_metrics.get('ev_ebitda')
        print(f"EV/EBITDA: {ev_ebitda:.2f}" if ev_ebitda else "EV/EBITDA: N/A")
        
        # Profitability metrics
        print(f"\n📈 METRICHE DI REDDITIVITÀ")
        print("-" * 40)
        
        gross_margin = self.fundamental_metrics.get('gross_margin')
        print(f"Margine Lordo: {gross_margin:.2f}%" if gross_margin else "Margine Lordo: N/A")
        
        operating_margin = self.fundamental_metrics.get('operating_margin')
        print(f"Margine Operativo: {operating_margin:.2f}%" if operating_margin else "Margine Operativo: N/A")
        
        profit_margin = self.fundamental_metrics.get('profit_margin')
        print(f"Margine di Profitto: {profit_margin:.2f}%" if profit_margin else "Margine di Profitto: N/A")
        
        roe = self.fundamental_metrics.get('roe')
        print(f"ROE: {roe:.2f}%" if roe else "ROE: N/A")
        
        roa = self.fundamental_metrics.get('roa')
        print(f"ROA: {roa:.2f}%" if roa else "ROA: N/A")
        
        # Financial health
        print(f"\n🏥 SALUTE FINANZIARIA")
        print("-" * 40)
        
        current_ratio = self.quality_metrics.get('current_ratio')
        print(f"Current Ratio: {current_ratio:.2f}" if current_ratio else "Current Ratio: N/A")
        
        debt_to_equity = self.quality_metrics.get('debt_to_equity')
        print(f"Debt-to-Equity: {debt_to_equity:.2f}" if debt_to_equity else "Debt-to-Equity: N/A")
        
        total_cash = self.quality_metrics.get('total_cash', 0)
        total_debt = self.quality_metrics.get('total_debt', 0)
        print(f"Liquidità Totale: ${total_cash/1_000_000:.1f}M" if total_cash > 0 else "Liquidità: N/A")
        print(f"Debito Totale: ${total_debt/1_000_000:.1f}M" if total_debt > 0 else "Debito Totale: N/A")
        
        # Growth metrics
        print(f"\n🚀 METRICHE DI CRESCITA")
        print("-" * 40)
        
        revenue_growth = self.growth_metrics.get('revenue_growth_yoy')
        print(f"Crescita Ricavi (YoY): {revenue_growth:.2f}%" if revenue_growth else "Crescita Ricavi: N/A")
        
        earnings_growth = self.growth_metrics.get('earnings_growth_yoy')
        print(f"Crescita Utili (YoY): {earnings_growth:.2f}%" if earnings_growth else "Crescita Utili: N/A")
        
        # Dividend information
        print(f"\n💎 INFORMAZIONI DIVIDENDI")
        print("-" * 40)
        
        dividend_yield = self.fundamental_metrics.get('dividend_yield', 0)
        dividend_rate = self.fundamental_metrics.get('dividend_rate', 0)
        
        if dividend_yield > 0:
            print(f"Dividend Yield: {dividend_yield:.2f}%")
            print(f"Dividend Rate: ${dividend_rate:.2f}")
            
            payout_ratio = self.fundamental_metrics.get('payout_ratio')
            print(f"Payout Ratio: {payout_ratio:.2f}%" if payout_ratio else "Payout Ratio: N/A")
        else:
            print("L'azienda non paga dividendi")
        
        # Investment recommendation
        recommendation = self.generate_investment_recommendation()
        print(f"\n🎯 RACCOMANDAZIONE DI INVESTIMENTO")
        print("-" * 40)
        print(f"Raccomandazione: {recommendation['color']} {recommendation['recommendation']}")
        print(f"Punteggio: {recommendation['score']}/100")
        print("\nFattori chiave:")
        for factor in recommendation['factors']:
            print(f"  {factor}")
        
        print("\n" + "=" * 80)
    
    def run_complete_analysis(self):
        """Esegue l'analisi fondamentale completa."""
        try:
            print(f"🔍 Avvio analisi fondamentale per {self.ticker}...")
            
            # Download data
            self.fetch_fundamental_data()
            
            # Calculate all metrics
            self.calculate_valuation_metrics()
            self.calculate_profitability_metrics()
            self.calculate_financial_health_metrics()
            self.calculate_growth_metrics()
            self.calculate_dividend_metrics()
            self.calculate_efficiency_metrics()
            
            # Print comprehensive analysis
            self.print_comprehensive_analysis()
            
            return True
            
        except Exception as e:
            print(f"❌ Errore durante l'analisi fondamentale: {str(e)}")
            return False


class ComparisonAnalyzer:
    """
    Classe per confrontare più ticker basati su analisi tecnica e fondamentale.
    """
    
    def __init__(self, tickers, period='1y', interval='1d', initial_capital=10000):
        self.tickers = [t.upper() for t in tickers]
        self.period = period
        self.interval = interval
        self.initial_capital = initial_capital
        self.technical_analyzers = {}
        self.fundamental_analyzers = {}
        self.comparison_data = {}
        self.correlation_matrix = None
    
    def run_technical_analysis(self):
        """Esegue l'analisi tecnica per tutti i ticker."""
        print("\n🔧 ANALISI TECNICA COMPARATIVA")
        print("-" * 50)
        
        for ticker in self.tickers:
            print(f"\nAnalizzando {ticker}...")
            try:
                analyzer = StockAnalyzer(ticker, self.period, self.interval, self.initial_capital)
                analyzer.fetch_data()
                analyzer.calculate_indicators()
                analyzer.backtest_strategy()
                analyzer.calculate_risk_metrics()
                analyzer.compare_with_benchmark()
                self.technical_analyzers[ticker] = analyzer
                self.comparison_data[ticker] = {
                    'total_return': analyzer.backtest_results['total_return'],
                    'sharpe_ratio': analyzer.backtest_results.get('sharpe_ratio', 0),
                    'sortino_ratio': analyzer.backtest_results.get('sortino_ratio', 0),
                    'volatility': analyzer.backtest_results.get('volatility', 0),
                    'max_drawdown': analyzer.backtest_results['max_drawdown'],
                    'beta': analyzer.backtest_results.get('beta', 0),
                    'alpha': analyzer.backtest_results.get('alpha', 0),
                    'win_rate': analyzer.backtest_results.get('win_rate', 0),
                    'profit_factor': analyzer.backtest_results.get('profit_factor', 0),
                    'avg_win': analyzer.backtest_results.get('avg_win', 0),
                    'avg_loss': analyzer.backtest_results.get('avg_loss', 0),
                    'max_consecutive_wins': analyzer.backtest_results.get('max_consecutive_wins', 0),
                    'max_consecutive_losses': analyzer.backtest_results.get('max_consecutive_losses', 0),
                    'calmar_ratio': analyzer.backtest_results.get('calmar_ratio', 0)
                }
                analyzer.print_results()
            except Exception as e:
                print(f"Errore nell'analisi tecnica per {ticker}: {e}")
    
    def calculate_correlation(self):
        """Calcola la matrice di correlazione tra i rendimenti dei ticker."""
        if not self.technical_analyzers:
            print("Nessun dato tecnico disponibile per calcolare la correlazione.")
            return
        
        returns_df = pd.DataFrame()
        for ticker, analyzer in self.technical_analyzers.items():
            daily_returns = analyzer.combined_df["Close"].pct_change().dropna()
            returns_df[ticker] = daily_returns
        
        # Calcola la matrice di correlazione
        self.correlation_matrix = returns_df.corr()
    
    def print_correlation_matrix(self):
        """Stampa la matrice di correlazione."""
        if self.correlation_matrix is None:
            print("Matrice di correlazione non calcolata.")
            return
        
        print("\n📈 MATRICE DI CORRELAZIONE TRA I RENDIMENTI")
        print("-" * 50)
        print(self.correlation_matrix.round(2))
    
    def plot_correlation_heatmap(self):
        """Visualizza la matrice di correlazione come heatmap."""
        if self.correlation_matrix is None:
            print("Matrice di correlazione non disponibile per il plot.")
            return
        
        plt.figure(figsize=(8, 6))
        plt.title("Heatmap Correlazione Rendimenti", color="white")
        sns.heatmap(self.correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, 
                    xticklabels=self.correlation_matrix.columns, yticklabels=self.correlation_matrix.columns,
                    cbar_kws={'label': 'Correlazione'})
        plt.xticks(color="white")
        plt.yticks(color="white")
        plt.gca().set_facecolor("black")
        plt.gcf().set_facecolor("#121212")
        plt.show()
    
    def run_fundamental_analysis(self):
        """Esegue l'analisi fondamentale per tutti i ticker."""
        print("\n📊 ANALISI FONDAMENTALE COMPARATIVA")
        print("-" * 50)
        
        for ticker in self.tickers:
            print(f"\nAnalizzando {ticker}...")
            try:
                analyzer = FundamentalAnalyzer(ticker)
                analyzer.run_complete_analysis()
                self.fundamental_analyzers[ticker] = analyzer
                
                # Aggiungi dati per confronto
                if ticker not in self.comparison_data:
                    self.comparison_data[ticker] = {}
                self.comparison_data[ticker].update({
                    'pe_ratio': analyzer.valuation_metrics.get('pe_ratio'),
                    'pb_ratio': analyzer.valuation_metrics.get('pb_ratio'),
                    'roe': analyzer.fundamental_metrics.get('roe'),
                    'debt_to_equity': analyzer.quality_metrics.get('debt_to_equity'),
                    'revenue_growth': analyzer.growth_metrics.get('revenue_growth_yoy'),
                    'dividend_yield': analyzer.fundamental_metrics.get('dividend_yield', 0),
                    'recommendation_score': analyzer.generate_investment_recommendation()['score']
                })
            except Exception as e:
                print(f"Errore nell'analisi fondamentale per {ticker}: {e}")
    
    def plot_technical_comparison(self):
        """Grafico comparativo dei prezzi e rendimenti."""
        if not self.technical_analyzers:
            print("Nessun dato tecnico disponibile per il confronto.")
            return
        
        plt.figure(figsize=(14, 10))
        
        # Subplot 1: Prezzi normalizzati
        ax1 = plt.subplot(2, 1, 1)
        ax1.set_title("Confronto Prezzi Normalizzati", color="white")
        for ticker, analyzer in self.technical_analyzers.items():
            normalized_price = analyzer.combined_df['Close'] / analyzer.combined_df['Close'].iloc[0]
            ax1.plot(analyzer.combined_df.index, normalized_price, label=ticker)
        ax1.legend()
        ax1.grid(True, color="#555555")
        ax1.set_facecolor("black")
        ax1.figure.set_facecolor("#121212")
        ax1.tick_params(axis="x", colors="white")
        ax1.tick_params(axis="y", colors="white")
        
        # Subplot 2: Cumulative Returns
        ax2 = plt.subplot(2, 1, 2)
        ax2.set_title("Confronto Rendimenti Cumulativi dal Backtest", color="white")
        for ticker, analyzer in self.technical_analyzers.items():
            cum_profit = analyzer.backtest_results['cumulative_profit'][1:]
            dates = analyzer.combined_df.index[1:]
            if len(dates) != len(cum_profit):
                min_len = min(len(dates), len(cum_profit))
                dates = dates[:min_len]
                cum_profit = cum_profit[:min_len]
            ax2.plot(dates, cum_profit, label=f"{ticker} ({analyzer.backtest_results['total_return']:.2f}%)")
        ax2.legend()
        ax2.grid(True, color="#555555")
        ax2.set_facecolor("black")
        ax2.figure.set_facecolor("#121212")
        ax2.tick_params(axis="x", colors="white")
        ax2.tick_params(axis="y", colors="white")
        
        plt.tight_layout()
        plt.show()
    
    def print_fundamental_comparison(self):
        """Stampa una tabella comparativa delle metriche fondamentali."""
        if not self.comparison_data:
            print("Nessun dato disponibile per il confronto.")
            return
        
        print("\n📊 CONFRONTO METRICHE FONDAMENTALI")
        print("-" * 80)
        
        # Header
        header = f"{'Ticker':<10} {'P/E':<8} {'P/B':<8} {'ROE':<8} {'D/E':<8} {'Rev Growth':<12} {'Div Yield':<12} {'Score':<8}"
        print(header)
        print("-" * len(header))
        
        # Data rows
        for ticker, data in self.comparison_data.items():
            pe = f"{data.get('pe_ratio', 'N/A'):.1f}" if data.get('pe_ratio') else 'N/A'
            pb = f"{data.get('pb_ratio', 'N/A'):.1f}" if data.get('pb_ratio') else 'N/A'
            roe = f"{data.get('roe', 'N/A'):.1f}" if data.get('roe') else 'N/A'
            de = f"{data.get('debt_to_equity', 'N/A'):.1f}" if data.get('debt_to_equity') else 'N/A'
            rev_growth = f"{data.get('revenue_growth', 'N/A'):.1f}%" if data.get('revenue_growth') else 'N/A'
            div_yield = f"{data.get('dividend_yield', 'N/A'):.1f}%" if data.get('dividend_yield') else 'N/A'
            score = f"{data.get('recommendation_score', 0):.0f}"
            
            row = f"{ticker:<10} {pe:<8} {pb:<8} {roe:<8} {de:<8} {rev_growth:<12} {div_yield:<12} {score:<8}"
            print(row)
        
        print("-" * len(header))
    
    def print_technical_comparison(self):
        """Stampa una tabella comparativa delle metriche tecniche."""
        if not self.comparison_data:
            print("Nessun dato disponibile per il confronto.")
            return
        
        print("\n🔧 CONFRONTO METRICHE TECNICHE")
        print("-" * 80)
        
        # Header
        header = f"{'Ticker':<10} {'Total Return':<14} {'Sharpe':<10} {'Sortino':<10} {'Volatility':<12} {'Max DD':<10} {'Beta':<8} {'Alpha':<8} {'Win Rate':<10} {'Profit Factor':<14} {'Avg Win':<10} {'Avg Loss':<10} {'Max Wins':<10} {'Max Losses':<12} {'Calmar':<8}"
        print(header)
        print("-" * len(header))
        
        # Data rows
        for ticker, data in self.comparison_data.items():
            total_return = f"{data.get('total_return', 0):.2f}%"
            sharpe = f"{data.get('sharpe_ratio', 0):.2f}"
            sortino = f"{data.get('sortino_ratio', 0):.2f}"
            vol = f"{data.get('volatility', 0):.2f}"
            max_dd = f"{data.get('max_drawdown', 0):.2f}"
            beta = f"{data.get('beta', 0):.2f}"
            alpha = f"{data.get('alpha', 0):.4f}"
            win_rate = f"{data.get('win_rate', 0):.2f}%"
            profit_factor = f"{data.get('profit_factor', 0):.2f}"
            avg_win = f"{data.get('avg_win', 0):.2f}"
            avg_loss = f"{data.get('avg_loss', 0):.2f}"
            max_wins = f"{data.get('max_consecutive_wins', 0)}"
            max_losses = f"{data.get('max_consecutive_losses', 0)}"
            calmar = f"{data.get('calmar_ratio', 0):.2f}"
            
            row = f"{ticker:<10} {total_return:<14} {sharpe:<10} {sortino:<10} {vol:<12} {max_dd:<10} {beta:<8} {alpha:<8} {win_rate:<10} {profit_factor:<14} {avg_win:<10} {avg_loss:<10} {max_wins:<10} {max_losses:<12} {calmar:<8}"
            print(row)
        
        print("-" * len(header))
    
    def run_comparison(self, analysis_type='both'):
        """Esegue il confronto completo."""
        if analysis_type in ['technical', 'both']:
            self.run_technical_analysis()
            self.calculate_correlation()
            self.print_correlation_matrix()
            self.plot_correlation_heatmap()
            self.print_technical_comparison()
            self.plot_technical_comparison()
        
        if analysis_type in ['fundamental', 'both']:
            self.run_fundamental_analysis()
            self.print_fundamental_comparison()


class PortfolioOptimizer:
    """
    Classe per l'ottimizzazione di portafoglio basata sulla teoria del portafoglio moderno (MPT).
    Include il modello Black-Litterman per incorporare viste soggettive.
    Calcola la frontiera efficiente e trova il portafoglio ottimale per massimizzare Sharpe ratio.
    """
    
    def __init__(self, tickers, period='2y', interval='1d', risk_free_rate=0.02, tau=0.05):
        self.tickers = [t.upper() for t in tickers]
        self.period = period
        self.interval = interval
        self.risk_free_rate = risk_free_rate
        self.tau = tau  # Parametro di rischio di mercato per Black-Litterman
        self.returns = None
        self.cov_matrix = None
        self.expected_returns = None
        self.bl_expected_returns = None  # Rendimenti attesi Black-Litterman
        self.optimal_weights = None
        self.efficient_frontier = None
        self.views = None  # Dizionario per viste BL
    
    def fetch_data(self):
        """Scarica i dati storici per tutti i ticker."""
        print(f"Downloading historical data for {', '.join(self.tickers)}...")
        data = {}
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=self.period, interval=self.interval)
                data[ticker] = hist['Close']
            except Exception as e:
                print(f"Error downloading data for {ticker}: {e}")
                return False
        self.price_data = pd.DataFrame(data)
        print("✓ Data downloaded successfully!")
        return True
    
    def calculate_returns_and_covariance(self):
        """Calcola rendimenti giornalieri e matrice di covarianza."""
        if self.price_data is None:
            print("No price data available.")
            return
        
        # Rendimenti giornalieri
        self.returns = self.price_data.pct_change().dropna()
        
        # Rendimenti attesi annualizzati (media dei rendimenti giornalieri * 252 giorni)
        self.expected_returns = self.returns.mean() * 252
        
        # Matrice di covarianza annualizzata
        self.cov_matrix = self.returns.cov() * 252
        
        print("✓ Returns and covariance matrix calculated!")
    
    def get_black_litterman_views(self):
        """Raccoglie le viste soggettive dall'utente per il modello Black-Litterman."""
        print("\n🔮 CONFIGURAZIONE VISTE BLACK-LITTERMAN")
        print("-" * 50)
        print("Inserisci le tue viste soggettive sui rendimenti attesi annuali.")
        print("Esempio: Se pensi che AAPL avrà un rendimento del 10% e MSFT del 5%, inserisci:")
        print("  - Asset coinvolti: AAPL,MSFT")
        print("  - Rendimento atteso: 0.10")
        print("  - Confidenza (0-1): 0.8")
        
        views = []
        while True:
            asset_input = input("Asset coinvolti (separati da virgola, o 'fine' per terminare): ").strip()
            if asset_input.lower() == 'fine':
                break
            assets = [a.strip().upper() for a in asset_input.split(',')]
            if not all(a in self.tickers for a in assets):
                print("Errore: Alcuni asset non sono nel portafoglio.")
                continue
            try:
                q = float(input("Rendimento atteso annuale (es. 0.10 per 10%): "))
                confidence = float(input("Confidenza nella vista (0-1, es. 0.8): "))
                if not (0 < confidence <= 1):
                    print("Confidenza deve essere tra 0 e 1.")
                    continue
                views.append({
                    'assets': assets,
                    'q': q,
                    'confidence': confidence
                })
            except ValueError:
                print("Input non valido. Riprova.")
        
        self.views = views
        print(f"✓ {len(views)} viste raccolte.")
    
    def calculate_black_litterman_returns(self):
        """Calcola i rendimenti attesi usando il modello Black-Litterman."""
        if self.expected_returns is None or self.cov_matrix is None:
            print("Calcola prima rendimenti e covarianza.")
            return
        
        if not self.views:
            print("Nessuna vista fornita. Usa rendimenti di mercato.")
            self.bl_expected_returns = self.expected_returns
            return
        
        n = len(self.tickers)
        tau_sigma = self.tau * self.cov_matrix.values
        
        # Matrice P (selezione degli asset per ogni vista)
        P = np.zeros((len(self.views), n))
        Q = np.zeros(len(self.views))
        Omega = np.zeros((len(self.views), len(self.views)))
        
        for i, view in enumerate(self.views):
            for asset in view['assets']:
                idx = self.tickers.index(asset)
                P[i, idx] = 1  # Vista assoluta su singolo asset; per viste relative, modificare
            Q[i] = view['q']
            # Omega diagonale con varianza basata su confidenza
            omega_val = (1 / view['confidence'] - 1) * np.dot(P[i], np.dot(tau_sigma, P[i].T))
            Omega[i, i] = omega_val
        
        # Formula Black-Litterman
        try:
            inv_tau_sigma = np.linalg.inv(tau_sigma)
            inv_omega = np.linalg.inv(Omega)
            A = inv_tau_sigma + np.dot(P.T, np.dot(inv_omega, P))
            B = np.dot(inv_tau_sigma, self.expected_returns.values) + np.dot(P.T, np.dot(inv_omega, Q))
            self.bl_expected_returns = pd.Series(np.dot(np.linalg.inv(A), B), index=self.tickers)
            print("✓ Rendimenti Black-Litterman calcolati.")
        except np.linalg.LinAlgError:
            print("Errore nel calcolo Black-Litterman. Usa rendimenti di mercato.")
            self.bl_expected_returns = self.expected_returns
    
    def portfolio_performance(self, weights, use_bl=False):
        """Calcola rendimento e volatilità del portafoglio."""
        expected_rets = self.bl_expected_returns if use_bl and self.bl_expected_returns is not None else self.expected_returns
        portfolio_return = np.dot(weights, expected_rets)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return portfolio_return, portfolio_volatility
    
    def negative_sharpe_ratio(self, weights, use_bl=False):
        """Funzione obiettivo per massimizzare Sharpe ratio (negativa per minimizzazione)."""
        portfolio_return, portfolio_volatility = self.portfolio_performance(weights, use_bl)
        return -(portfolio_return - self.risk_free_rate) / portfolio_volatility
    
    def minimize_volatility(self, target_return, use_bl=False):
        """Minimizza la volatilità per un rendimento target."""
        expected_rets = self.bl_expected_returns if use_bl and self.bl_expected_returns is not None else self.expected_returns
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Somma pesi = 1
            {'type': 'eq', 'fun': lambda x: self.portfolio_performance(x, use_bl)[0] - target_return}  # Rendimento target
        ]
        bounds = tuple((0, 1) for _ in range(len(self.tickers)))  # Pesi tra 0 e 1
        initial_guess = np.array([1/len(self.tickers)] * len(self.tickers))  # Guess iniziale equi-pesato
        
        result = opt.minimize(
            lambda x: self.portfolio_performance(x, use_bl)[1],  # Minimizza volatilità
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        return result.x if result.success else None
    
    def optimize_portfolio(self, use_bl=False):
        """Ottimizza il portafoglio per massimizzare Sharpe ratio."""
        expected_rets = self.bl_expected_returns if use_bl and self.bl_expected_returns is not None else self.expected_returns
        if expected_rets is None or self.cov_matrix is None:
            print("Calculate returns and covariance first.")
            return
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Somma pesi = 1
        bounds = tuple((0, 1) for _ in range(len(self.tickers)))  # Pesi tra 0 e 1
        initial_guess = np.array([1/len(self.tickers)] * len(self.tickers))  # Guess iniziale equi-pesato
        
        result = opt.minimize(
            lambda w: self.negative_sharpe_ratio(w, use_bl),
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            self.optimal_weights = result.x
            optimal_return, optimal_volatility = self.portfolio_performance(self.optimal_weights, use_bl)
            optimal_sharpe = (optimal_return - self.risk_free_rate) / optimal_volatility
            
            print("✓ Portfolio optimized successfully!")
            print(f"Optimal Sharpe Ratio: {optimal_sharpe:.2f}")
            print(f"Expected Annual Return: {optimal_return:.2f}")
            print(f"Annual Volatility: {optimal_volatility:.2f}")
            print("\nOptimal Weights:")
            for ticker, weight in zip(self.tickers, self.optimal_weights):
                print(f"  {ticker}: {weight:.2f} ({weight*100:.1f}%)")
            
            return self.optimal_weights, optimal_return, optimal_volatility, optimal_sharpe
        else:
            print("❌ Optimization failed.")
            return None
    
    def calculate_efficient_frontier(self, num_portfolios=100, use_bl=False):
        """Calcola punti della frontiera efficiente."""
        expected_rets = self.bl_expected_returns if use_bl and self.bl_expected_returns is not None else self.expected_returns
        if expected_rets is None or self.cov_matrix is None:
            print("Calculate returns and covariance first.")
            return
        
        target_returns = np.linspace(expected_rets.min(), expected_rets.max(), num_portfolios)
        efficient_portfolios = []
        
        for target_return in target_returns:
            weights = self.minimize_volatility(target_return, use_bl)
            if weights is not None:
                ret, vol = self.portfolio_performance(weights, use_bl)
                efficient_portfolios.append((ret, vol, weights))
        
        self.efficient_frontier = efficient_portfolios
        print("✓ Efficient frontier calculated!")
        return self.efficient_frontier
    
    def plot_efficient_frontier(self, use_bl=False):
        """Visualizza la frontiera efficiente."""
        if self.efficient_frontier is None:
            print("Calculate efficient frontier first.")
            return
        
        returns = [p[0] for p in self.efficient_frontier]
        volatilities = [p[1] for p in self.efficient_frontier]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(volatilities, returns, c=returns, cmap='viridis', marker='o', s=10, alpha=0.7)
        plt.colorbar(label='Expected Return')
        plt.xlabel('Volatility (Risk)')
        plt.ylabel('Expected Return')
        title = 'Efficient Frontier (Black-Litterman)' if use_bl else 'Efficient Frontier'
        plt.title(title, color='white')
        
        # Plot portafoglio ottimale
        if self.optimal_weights is not None:
            opt_ret, opt_vol = self.portfolio_performance(self.optimal_weights, use_bl)
            plt.scatter(opt_vol, opt_ret, color='red', marker='*', s=200, label='Optimal Portfolio (Max Sharpe)')
            plt.legend()
        
        # Plot singoli asset
        for i, ticker in enumerate(self.tickers):
            plt.scatter(np.sqrt(self.cov_matrix.iloc[i, i]), self.expected_returns[i], 
                       marker='x', s=50, label=ticker)
        
        plt.grid(True, color="#555555")
        plt.gca().set_facecolor("black")
        plt.gcf().set_facecolor("#121212")
        plt.tick_params(axis="x", colors="white")
        plt.tick_params(axis="y", colors="white")
        plt.legend()
        plt.show()
    
    def print_portfolio_summary(self, use_bl=False):
        """Stampa un riassunto del portafoglio ottimale."""
        if self.optimal_weights is None:
            print("Optimize portfolio first.")
            return
        
        print("\n📊 RIASSUNTO PORTAFOGLIO OTTIMIZZATO")
        print("-" * 50)
        
        optimal_return, optimal_volatility = self.portfolio_performance(self.optimal_weights, use_bl)
        optimal_sharpe = (optimal_return - self.risk_free_rate) / optimal_volatility
        
        print(f"Rendimento Annuale Atteso: {optimal_return:.2f}")
        print(f"Volatilità Annuale: {optimal_volatility:.2f}")
        print(f"Sharpe Ratio: {optimal_sharpe:.2f}")
        print(f"Tasso Risk-Free: {self.risk_free_rate:.2f}")
        
        print("\nAllocazione Ottimale:")
        for ticker, weight in zip(self.tickers, self.optimal_weights):
            print(f"  {ticker}: {weight*100:.1f}%")
    
    def run_portfolio_optimization(self, use_bl=False):
        """Esegue l'ottimizzazione completa del portafoglio."""
        print(f"🚀 AVVIO OTTIMIZZAZIONE PORTAFOGLIO PER {', '.join(self.tickers)}")
        print("-" * 60)
        
        if not self.fetch_data():
            return
        
        self.calculate_returns_and_covariance()
        
        if use_bl:
            self.get_black_litterman_views()
            self.calculate_black_litterman_returns()
        
        # Ottimizza per Sharpe ratio massimo
        self.optimize_portfolio(use_bl)
        
        # Calcola frontiera efficiente
        self.calculate_efficient_frontier(use_bl=use_bl)
        
        # Visualizza risultati
        self.plot_efficient_frontier(use_bl)
        self.print_portfolio_summary(use_bl)
        
        print("✓ Portfolio optimization completed!")



class PredictionAnalyzer:
    """
    Classe avanzata per previsioni sui prezzi delle azioni utilizzando modelli statistici e ML.
    Include tuning, ensemble e previsioni multi-step.
    """
    
    def __init__(self, data, ticker, forecast_days=30):
        self.data = data  # DataFrame con prezzi e indicatori (da StockAnalyzer)
        self.ticker = ticker
        self.forecast_days = forecast_days
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.ensemble_predictions = None
    
    def prepare_data(self, model_type='arima'):
        """Prepara dati avanzati con features tecniche."""
        # Aggiungi indicatori tecnici se non presenti
        if 'RSI' not in self.data.columns:
            self.data['RSI'] = ta.RSI(self.data['Close'], timeperiod=14)
        if 'MACD' not in self.data.columns:
            macd, _, _ = ta.MACD(self.data['Close'])
            self.data['MACD'] = macd
        if 'Upper Band' not in self.data.columns:
            upper, _, lower = ta.BBANDS(self.data['Close'])
            self.data['Upper Band'] = upper
            self.data['Lower Band'] = lower
        self.data['Volume'] = self.data.get('Volume', 0)  # Assumi Volume presente
        
        if model_type in ['arima', 'sarima', 'prophet']:
            self.train_data = self.data['Close'].dropna()
        elif model_type in ['linear', 'rf', 'xgb']:
            self.train_data = self._create_advanced_features()
        elif model_type == 'lstm':
            self.train_data, self.scaler = self._prepare_lstm_data()
    
    def _create_advanced_features(self, lags=5):
        """Crea features avanzate con lags e indicatori (escludi Close lags per evitare leakage)."""
        df = self.data.copy()
        for lag in range(1, lags + 1):
            # Rimuovi Close_lag per evitare overfitting in modelli lineari
            # df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'RSI_lag_{lag}'] = df['RSI'].shift(lag)
            df[f'MACD_lag_{lag}'] = df['MACD'].shift(lag)
        df['Volume_pct_change'] = df['Volume'].pct_change()
        df.dropna(inplace=True)
        return df
    
    def _prepare_lstm_data(self, lookback=60):
        """Prepara sequenze LSTM con features multiple."""
        features = ['Close', 'RSI', 'MACD', 'Upper Band', 'Lower Band', 'Volume']
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data[features].dropna().values)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i, 0])  # Prevedi Close
        
        X = np.array(X)
        y = np.array(y)
        return (X, y), scaler
    
    def train_arima(self, order=(5,1,0)):
        """Addestra ARIMA con auto-tuning semplificato."""
        print(f"Training ARIMA model for {self.ticker}...")
        try:
            model = ARIMA(self.train_data, order=order)
            self.models['arima'] = model.fit()
            print("✓ ARIMA model trained!")
        except:
            print("❌ ARIMA training failed.")
    
    def train_sarima(self, order=(5,1,0), seasonal_order=(1,1,1,12)):
        """Addestra SARIMA."""
        print(f"Training SARIMA model for {self.ticker}...")
        try:
            model = SARIMAX(self.train_data, order=order, seasonal_order=seasonal_order)
            self.models['sarima'] = model.fit(disp=False)
            print("✓ SARIMA model trained!")
        except:
            print("❌ SARIMA training failed.")
    
    def train_prophet(self):
        """Addestra Prophet."""
        from prophet import Prophet  # Assumi installato
        print(f"Training Prophet model for {self.ticker}...")
        try:
            df_prophet = pd.DataFrame({'ds': self.train_data.index, 'y': self.train_data.values})
            model = Prophet()
            model.fit(df_prophet)
            self.models['prophet'] = model
            print("✓ Prophet model trained!")
        except:
            print("❌ Prophet training failed.")
    

    def train_linear_regression(self):
        """Addestra Regressione Lineare con cross-validation."""
        print(f"Training Linear Regression model for {self.ticker}...")
        X = self.train_data.drop('Close', axis=1)
        y = self.train_data['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = Ridge(alpha=0.1)  # Use Ridge instead of LinearRegression for regularization
        model.fit(X_train, y_train)
        self.models['linear'] = model
        
        y_pred = model.predict(X_test)
        self._calculate_metrics(y_test, y_pred, 'linear')
        print("✓ Linear Regression model trained!")
    
    def train_random_forest(self):
        """Addestra Random Forest con GridSearch."""
        from sklearn.model_selection import GridSearchCV
        print(f"Training Random Forest model for {self.ticker}...")
        X = self.train_data.drop('Close', axis=1)
        y = self.train_data['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
        grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3)
        grid.fit(X_train, y_train)
        self.models['rf'] = grid.best_estimator_
        
        y_pred = self.models['rf'].predict(X_test)
        self._calculate_metrics(y_test, y_pred, 'rf')
        print("✓ Random Forest model trained!")
    
    def train_xgboost(self):
        """Addestra XGBoost con tuning."""
        import xgboost as xgb
        print(f"Training XGBoost model for {self.ticker}...")
        X = self.train_data.drop('Close', axis=1)
        y = self.train_data['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        model.fit(X_train, y_train)
        self.models['xgb'] = model
        
        y_pred = model.predict(X_test)
        self._calculate_metrics(y_test, y_pred, 'xgb')
        print("✓ XGBoost model trained!")
    
    def train_lstm(self, epochs=50, batch_size=32):
        """Addestra LSTM con tuning."""
        print(f"Training LSTM model for {self.ticker}...")
        X, y = self.train_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        self.models['lstm'] = model
        
        y_pred = model.predict(X_test).flatten()
        # Fix: Use the scaler's parameters for the first feature ('Close') to inverse transform
        y_test_inv = y_test * self.scaler.scale_[0] + self.scaler.mean_[0]
        y_pred_inv = y_pred * self.scaler.scale_[0] + self.scaler.mean_[0]
        self._calculate_metrics(y_test_inv, y_pred_inv, 'lstm')
        print("✓ LSTM model trained!")
    
    def _calculate_metrics(self, y_true, y_pred, model_name):
        """Calcola metriche avanzate."""
        self.metrics[model_name] = {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def forecast(self, model_type):
        """Effettua previsioni multi-step."""
        if model_type not in self.models:
            return
        
        print(f"Forecasting with {model_type.upper()} for {self.forecast_days} days...")
        
        if model_type in ['arima', 'sarima']:
            forecast = self.models[model_type].forecast(steps=self.forecast_days)
            self.predictions[model_type] = forecast
        elif model_type == 'prophet':
            future = self.models['prophet'].make_future_dataframe(periods=self.forecast_days)
            forecast = self.models['prophet'].predict(future)
            self.predictions['prophet'] = forecast['yhat'].tail(self.forecast_days).values
        elif model_type in ['linear', 'rf', 'xgb']:
            self.predictions[model_type] = self._multi_step_forecast_ml(model_type)
        elif model_type == 'lstm':
            self.predictions['lstm'] = self._multi_step_forecast_lstm()
        
        print(f"✓ Forecast completed for {model_type.upper()}!")
    

    def _multi_step_forecast_ml(self, model_type):
        """Improved multi-step forecast for ML models."""
        preds = []
        # Start with last known features
        last_row = self.train_data.drop('Close', axis=1).iloc[-1:].copy()
        for _ in range(self.forecast_days):
            pred = self.models[model_type].predict(last_row)[0]
            preds.append(pred)
            # Simplified: Use the same last_row for all steps (features remain static)
            # Do not update lags to avoid feature mismatch
        return np.array(preds)
    



    def _multi_step_forecast_lstm(self):
        """Previsioni multi-step per LSTM."""
        preds = []
        last_sequence = self.train_data[0][-1:].copy()
        for _ in range(self.forecast_days):
            pred = self.models['lstm'].predict(last_sequence)[0][0]
            # Fix: Use the scaler's parameters for the first feature ('Close') to inverse transform
            preds.append(pred * self.scaler.scale_[0] + self.scaler.mean_[0])
            # Aggiorna sequenza
            new_row = np.array([pred, 0, 0, 0, 0, 0]).reshape(1, 1, -1)
            last_sequence = np.concatenate([last_sequence[:, 1:], new_row], axis=1)
        return np.array(preds)

    
    def create_ensemble(self):
        """Crea previsioni ensemble mediando modelli disponibili."""
        available_preds = [self.predictions[m] for m in ['rf', 'xgb', 'lstm'] if m in self.predictions]
        if available_preds:
            self.ensemble_predictions = np.mean(available_preds, axis=0)
            print("✓ Ensemble predictions created!")
    
    def backtest_predictions(self, model_type, test_days=30):
        """Backtest using training test set instead of invalid future comparison."""
        if model_type not in self.metrics:
            return
        # Use metrics from training test set
        mape = self.metrics[model_type]['mape']
        print(f"Backtest MAPE for {model_type} (training test set): {mape:.2f}%")
    
    def plot_forecast(self, model_type):
        """Visualizza previsioni con intervallo di confidenza."""
        if model_type not in self.predictions:
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data['Close'], label='Historical Prices', color='blue')
        
        forecast_dates = pd.date_range(start=self.data.index[-1], periods=self.forecast_days + 1, freq='D')[1:]
        plt.plot(forecast_dates, self.predictions[model_type], label=f'{model_type.upper()} Forecast', color='red', linestyle='--')
        
        # Aggiungi ensemble se disponibile
        if self.ensemble_predictions is not None:
            plt.plot(forecast_dates, self.ensemble_predictions, label='Ensemble Forecast', color='green', linestyle='-.')
        
        plt.title(f'Advanced Price Forecast for {self.ticker}', color='white')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, color="#555555")
        plt.gca().set_facecolor("black")
        plt.gcf().set_facecolor("#121212")
        plt.tick_params(axis="x", colors="white")
        plt.tick_params(axis="y", colors="white")
        plt.show()
    
    def print_metrics(self):
        """Stampa metriche estese."""
        print("\n📊 MODEL EVALUATION METRICS")
        print("-" * 50)
        for model, mets in self.metrics.items():
            print(f"{model.upper()}: MSE={mets['mse']:.4f}, MAE={mets['mae']:.4f}, RMSE={mets['rmse']:.4f}, MAPE={mets['mape']:.2f}%")
    
    def run_all_models(self):
        """Addestra, prevede e valuta tutti i modelli avanzati."""
        models_to_run = ['arima', 'sarima', 'prophet', 'linear', 'rf', 'xgb', 'lstm']
        for model in models_to_run:
            self.prepare_data(model)
            if model == 'arima':
                self.train_arima()
                self.forecast('arima')
            elif model == 'sarima':
                self.train_sarima()
                self.forecast('sarima')
            elif model == 'prophet':
                self.train_prophet()
                self.forecast('prophet')
            elif model == 'linear':
                self.train_linear_regression()
                self.forecast('linear')
            elif model == 'rf':
                self.train_random_forest()
                self.forecast('rf')
            elif model == 'xgb':
                self.train_xgboost()
                self.forecast('xgb')
            elif model == 'lstm':
                self.train_lstm()
                self.forecast('lstm')
        
        self.create_ensemble()
        self.print_metrics()
        
        # Plot e backtest
        for model in models_to_run:
            if model in self.predictions:
                self.plot_forecast(model)
                self.backtest_predictions(model)

# Modifica al blocco main per supportare più ticker
if __name__ == "__main__":
    print("=== STOCK ANALYSIS SUITE ===\n")
    
    # Input per più ticker
    tickers_input = input("Inserisci i ticker symbol (separati da virgola, es. AAPL,MSFT,GOOGL): ")
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    
    if not tickers:
        print("Nessun ticker valido inserito.")
        exit()
    
    print(f"\nTicker selezionati: {', '.join(tickers)}")
    
    print("\nScegli il tipo di analisi:")
    print("1. Analisi Tecnica Completa")
    print("2. Analisi Fondamentale Completa")
    print("3. Analisi Combinata (Tecnica + Fondamentale)")
    print("4. Confronto Multi-Ticker (Tecnica)")
    print("5. Confronto Multi-Ticker (Fondamentale)")
    print("6. Confronto Multi-Ticker (Entrambi)")
    print("7. Ottimizzazione Portafoglio (MPT)")
    print("8. Ottimizzazione Portafoglio (Black-Litterman)")
    print("9. Previsioni sui Prezzi (Multi-Modello)")
    
    choice = input("\nInserisci la tua scelta (1-9): ")
    
    if choice in ['1', '3']:
        # Technical Analysis per singolo ticker (se uno solo)
        if len(tickers) == 1:
            ticker = tickers[0]
            print(f"\n🔧 ANALISI TECNICA - {ticker}")
            print("-" * 50)
            
            period = input("Inserisci il periodo (es. '1y' per 1 anno, '6mo' per 6 mesi): ")
            interval = input("Inserisci l'intervallo dei dati (es. '1d' per giornaliero, '1wk' per settimanale): ")
            initial_capital = int(input("Inserisci il capitale iniziale per il backtesting (es. 10000): "))
            
            technical_analyzer = StockAnalyzer(ticker, period, interval, initial_capital)
            technical_analyzer.run()
        else:
            print("Per analisi tecnica singola, inserisci un solo ticker.")
    
    if choice in ['2', '3']:
        # Fundamental Analysis per singolo ticker (se uno solo)
        if len(tickers) == 1:
            ticker = tickers[0]
            print(f"\n📊 ANALISI FONDAMENTALE - {ticker}")
            print("-" * 50)
            
            fundamental_analyzer = FundamentalAnalyzer(ticker)
            fundamental_analyzer.run_complete_analysis()
        else:
            print("Per analisi fondamentale singola, inserisci un solo ticker.")
    
    if choice in ['4', '5', '6']:
        # Confronto multi-ticker
        period = input("Inserisci il periodo (es. '1y'): ") if choice in ['4', '6'] else '1y'
        interval = input("Inserisci l'intervallo (es. '1d'): ") if choice in ['4', '6'] else '1d'
        initial_capital = int(input("Inserisci il capitale iniziale (es. 10000): ")) if choice in ['4', '6'] else 10000
        
        comparison_analyzer = ComparisonAnalyzer(tickers, period, interval, initial_capital)
        
        if choice == '4':
            comparison_analyzer.run_comparison('technical')
        elif choice == '5':
            comparison_analyzer.run_comparison('fundamental')
        elif choice == '6':
            comparison_analyzer.run_comparison('both')
    
    if choice in ['7', '8']:
        # Ottimizzazione Portafoglio
        period = input("Inserisci il periodo storico per l'ottimizzazione (es. '2y'): ")
        risk_free_rate = float(input("Inserisci il tasso risk-free (es. 0.02 per 2%): "))
        
        use_bl = (choice == '8')
        portfolio_optimizer = PortfolioOptimizer(tickers, period, risk_free_rate=risk_free_rate)
        portfolio_optimizer.run_portfolio_optimization(use_bl=use_bl)
    
    if choice == '9':
        # Previsioni sui prezzi per singolo ticker
        if len(tickers) == 1:
            ticker = tickers[0]
            print(f"\n🔮 PREVISIONI PREZZI - {ticker}")
            print("-" * 50)
            
            period = input("Inserisci il periodo storico (es. '2y'): ")
            forecast_days = int(input("Inserisci il numero di giorni per le previsioni (es. 30): "))
            
            # Scarica dati
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval='1d')
            
            # Crea analyzer
            predictor = PredictionAnalyzer(data, ticker, forecast_days)
            predictor.run_all_models()
        else:
            print("Per previsioni, inserisci un solo ticker.")
    
    print(f"\n✅ Analisi completata per {', '.join(tickers)}!")
