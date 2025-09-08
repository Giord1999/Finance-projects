import pandas as pd 
import yfinance as yf 
import pandas_datareader as pdr
import matplotlib.pyplot as plt 
import datetime as dt 
import mplfinance as mpf
import numpy as np

ticker = str(input("Enter the ticker symbol: "))
stock = yf.Ticker(ticker)
data = stock.history(period="1y", interval="1d")

#Calcoliamo la differenza tra una riga e quella precedente e controlliamo se questa differenza è positiva o 
#negativa
delta = data["Close"].diff(1)   
delta.dropna(inplace=True)

positive = delta.copy()
negative = delta.copy()

## Controlliamo e separiamo tutti i dati che non sono negativi
positive[positive<0] = 0 
negative[negative>0] = 0

days = 14 
avg_gain = positive.rolling(window=days).mean()
avg_loss = abs(negative.rolling(window=days).mean())

relative_strenght = avg_gain/avg_loss
RSI = 100.0 -(100.0/(1.0+relative_strenght))
combined_df = pd.DataFrame()
combined_df['Close'] = data["Close"]
combined_df["RSI"] = RSI

# Calculate MACD
exp12 = data["Close"].ewm(span=12, adjust=False).mean()
exp26 = data["Close"].ewm(span=26, adjust=False).mean()
macd = exp12 - exp26
signal = macd.ewm(span=9, adjust=False).mean()
histogram = macd - signal
combined_df["MACD"] = macd
combined_df["Signal Line"] = signal
combined_df["Histogram"] = histogram

# Calculate Bollinger Bands
combined_df["MA20"] = data["Close"].rolling(window=20).mean()
combined_df["Upper Band"] = combined_df["MA20"] + 2 * data["Close"].rolling(window=20).std()
combined_df["Lower Band"] = combined_df["MA20"] - 2 * data["Close"].rolling(window=20).std()


# Calculate Stochastic Oscillator
low_min = data["Low"].rolling(window=14).min()
high_max = data["High"].rolling(window=14).max()
combined_df["%K"] = 100 * ((data["Close"] - low_min) / (high_max - low_min))
combined_df["%D"] = combined_df["%K"].rolling(window=3).mean()

# Calculate Williams %R
combined_df["Williams %R"] = -100 * ((high_max - data["Close"]) / (high_max - low_min))

# Calculate CCI
typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
sma_tp = typical_price.rolling(window=20).mean()
mad_tp = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=False)
combined_df["CCI"] = (typical_price - sma_tp) / (0.015 * mad_tp)

# Calculate ADX
high_diff = data["High"].diff()
low_diff = data["Low"].diff()
plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
tr = np.maximum(data["High"] - data["Low"], np.maximum(abs(data["High"] - data["Close"].shift(1)), abs(data["Low"] - data["Close"].shift(1))))
atr = pd.Series(tr).rolling(window=14).mean()
plus_di = 100 * (pd.Series(plus_dm).rolling(window=14).mean() / atr)
minus_di = 100 * (pd.Series(minus_dm).rolling(window=14).mean() / atr)
dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
combined_df["ADX"] = dx.rolling(window=14).mean()

# Calculate OBV
obv = [0]
for i in range(1, len(data)):
    if data["Close"][i] > data["Close"][i-1]:
        obv.append(obv[-1] + data["Volume"][i])
    elif data["Close"][i] < data["Close"][i-1]:
        obv.append(obv[-1] - data["Volume"][i])
    else:
        obv.append(obv[-1])
combined_df["OBV"] = obv


# Calculate Ichimoku Cloud
high9 = data["High"].rolling(window=9).max()
low9 = data["Low"].rolling(window=9).min()
combined_df["Tenkan-sen"] = (high9 + low9) / 2

high26 = data["High"].rolling(window=26).max()
low26 = data["Low"].rolling(window=26).min()
combined_df["Kijun-sen"] = (high26 + low26) / 2

combined_df["Senkou Span A"] = ((combined_df["Tenkan-sen"] + combined_df["Kijun-sen"]) / 2).shift(26)

high52 = data["High"].rolling(window=52).max()
low52 = data["Low"].rolling(window=52).min()
combined_df["Senkou Span B"] = ((high52 + low52) / 2).shift(26)

combined_df["Chikou Span"] = data["Close"].shift(-26)

# Calculate Parabolic SAR
# Note: This is a simplified implementation; for accuracy, consider using TA-Lib
sar = [data["Low"][0]]
af = 0.02
ep = data["High"][0]
af_increment = 0.02
max_af = 0.2

for i in range(1, len(data)):
    sar.append(sar[-1] + af * (ep - sar[-1]))
    if data["High"][i] > ep:
        ep = data["High"][i]
        af = min(af + af_increment, max_af)
    elif data["Low"][i] < ep:
        ep = data["Low"][i]
        af = min(af + af_increment, max_af)
    if (data["Close"][i] > sar[-1] and data["Close"][i-1] < sar[-1]) or (data["Close"][i] < sar[-1] and data["Close"][i-1] > sar[-1]):
        af = 0.02
        ep = data["High"][i] if data["Close"][i] > sar[-1] else data["Low"][i]

combined_df["Parabolic SAR"] = sar

# Calculate Fibonacci Retracements
# Using the last 52-week high and low for simplicity
fib_high = data["High"].max()
fib_low = data["Low"].min()
fib_range = fib_high - fib_low
fib_levels = {
    "0%": fib_high,
    "23.6%": fib_high - 0.236 * fib_range,
    "38.2%": fib_high - 0.382 * fib_range,
    "50%": fib_high - 0.5 * fib_range,
    "61.8%": fib_high - 0.618 * fib_range,
    "100%": fib_low
}

#Plot del relative strenght index
plt.figure(figsize=(12,18))  # Increased height for more subplots
ax1 = plt.subplot(611)
ax1.set_title("Stock Price with Bollinger Bands for {}".format(ticker), color="white")   #Titolo del grafico
ax1.plot(combined_df.index, combined_df["Close"], color = "lightgray", label="Close")
ax1.plot(combined_df.index, combined_df["MA20"], color="blue", label="MA20")
ax1.plot(combined_df.index, combined_df["Upper Band"], color="red", label="Upper Band")
ax1.plot(combined_df.index, combined_df["Lower Band"], color="green", label="Lower Band")
ax1.legend()
ax1.grid(True, color = "#555555")
ax1.set_axisbelow(True)
ax1.set_facecolor("black")
ax1.figure.set_facecolor("#121212")
ax1.tick_params(axis="x", colors="white")
ax1.tick_params(axis="y", colors="white")

ax2 = plt.subplot(612, sharex=ax1)
ax2.set_title("RSI value for {}".format(ticker), color="white")   #Titolo del grafico
ax2.plot(combined_df.index, combined_df["RSI"], color = "lightgray")
ax2.axhline(0, linestyle="--", alpha=0.5, color = "#ff0000")
ax2.axhline(10, linestyle="--", alpha=0.5, color = "#ffaa00")
ax2.axhline(20, linestyle="--", alpha=0.5, color = "#00ff00")
ax2.axhline(30, linestyle="--", alpha=0.5, color = "#cccccc")
ax2.axhline(70, linestyle="--", alpha=0.5, color = "#cccccc")
ax2.axhline(80, linestyle="--", alpha=0.5, color = "#00ff00")
ax2.axhline(90, linestyle="--", alpha=0.5, color = "#ffaa00")
ax2.axhline(100, linestyle="--", alpha=0.5, color = "#ff0000")

ax2.grid(False)
ax2.set_axisbelow(True)
ax2.set_facecolor("black")
ax2.tick_params(axis="x", colors="white")
ax2.tick_params(axis="y", colors="white")

ax3 = plt.subplot(613, sharex=ax1)
ax3.set_title("MACD for {}".format(ticker), color="white")
ax3.plot(combined_df.index, combined_df["MACD"], color="blue", label="MACD")
ax3.plot(combined_df.index, combined_df["Signal Line"], color="red", label="Signal Line")
ax3.bar(combined_df.index, combined_df["Histogram"], color="gray", alpha=0.5, label="Histogram")
ax3.legend()
ax3.grid(True, color="#555555")
ax3.set_axisbelow(True)
ax3.set_facecolor("black")
ax3.tick_params(axis="x", colors="white")
ax3.tick_params(axis="y", colors="white")

ax4 = plt.subplot(614, sharex=ax1)
ax4.set_title("Stochastic Oscillator for {}".format(ticker), color="white")
ax4.plot(combined_df.index, combined_df["%K"], color="blue", label="%K")
ax4.plot(combined_df.index, combined_df["%D"], color="red", label="%D")
ax4.axhline(20, linestyle="--", alpha=0.5, color="#cccccc")
ax4.axhline(80, linestyle="--", alpha=0.5, color="#cccccc")
ax4.legend()
ax4.grid(True, color="#555555")
ax4.set_axisbelow(True)
ax4.set_facecolor("black")
ax4.tick_params(axis="x", colors="white")
ax4.tick_params(axis="y", colors="white")

ax5 = plt.subplot(615, sharex=ax1)
ax5.set_title("Williams %R and CCI for {}".format(ticker), color="white")
ax5.plot(combined_df.index, combined_df["Williams %R"], color="green", label="Williams %R")
ax5.plot(combined_df.index, combined_df["CCI"], color="orange", label="CCI")
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
ax6.set_title("ADX and OBV for {}".format(ticker), color="white")
ax6.plot(combined_df.index, combined_df["ADX"], color="purple", label="ADX")
ax6.axhline(25, linestyle="--", alpha=0.5, color="#cccccc")
ax6.legend()
ax6.grid(True, color="#555555")
ax6.set_axisbelow(True)
ax6.set_facecolor("black")
ax6.tick_params(axis="x", colors="white")
ax6.tick_params(axis="y", colors="white")

# Secondary y-axis for OBV
ax6_twin = ax6.twinx()
ax6_twin.plot(combined_df.index, combined_df["OBV"], color="cyan", label="OBV")
ax6_twin.tick_params(axis="y", colors="white")
ax6_twin.legend(loc="upper right")

plt.tight_layout()
plt.show()
#Plot del candlestick graph
mpf.plot(data=data, type="candle", style="yahoo", volume=True, title="Candlestick chart for {}".format(ticker), mav=(20,50), figratio=(12,8), figscale=1.5)


# New separate figure for Ichimoku, Parabolic SAR, and Fibonacci
plt.figure(figsize=(12, 8))
ax_new = plt.subplot(111)
ax_new.set_title("Price with Ichimoku Cloud, Parabolic SAR, and Fibonacci Retracements for {}".format(ticker), color="white")
ax_new.plot(combined_df.index, combined_df["Close"], color="lightgray", label="Close")

# Ichimoku Cloud
ax_new.plot(combined_df.index, combined_df["Tenkan-sen"], color="blue", label="Tenkan-sen")
ax_new.plot(combined_df.index, combined_df["Kijun-sen"], color="red", label="Kijun-sen")
ax_new.fill_between(combined_df.index, combined_df["Senkou Span A"], combined_df["Senkou Span B"], where=combined_df["Senkou Span A"] >= combined_df["Senkou Span B"], color="green", alpha=0.3, label="Cloud (Bullish)")
ax_new.fill_between(combined_df.index, combined_df["Senkou Span A"], combined_df["Senkou Span B"], where=combined_df["Senkou Span A"] < combined_df["Senkou Span B"], color="red", alpha=0.3, label="Cloud (Bearish)")
ax_new.plot(combined_df.index, combined_df["Chikou Span"], color="purple", label="Chikou Span")

# Parabolic SAR
ax_new.scatter(combined_df.index, combined_df["Parabolic SAR"], color="orange", s=1, label="Parabolic SAR")

# Fibonacci Retracements
for level, value in fib_levels.items():
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
