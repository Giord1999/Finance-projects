import datetime as dt 
import matplotlib.pyplot as plt 
import yfinance as yf 
import pandas_datareader as pdr 

ma_1 = 30 
ma_2 = 100
ticker = "META"
stock = yf.Ticker(ticker)
data = stock.history(period="max", interval="1d")
data[f"SMA_{ma_1}"] = data["Close"].rolling(window=ma_1).mean()
data[f"SMA_{ma_2}"] = data["Close"].rolling(window=ma_2).mean()

data = data.iloc[ma_2:]
buy_signals = []
sell_signals = []
trigger = 0

for x in range(len(data)):
    if data[f"SMA_{ma_1}"].iloc[x] > data[f"SMA_{ma_2}"].iloc[x] and trigger!=1:
        buy_signals.append(data["Close"].iloc[x])
        sell_signals.append(float("nan"))
        trigger = 1
    elif data[f"SMA_{ma_1}"].iloc[x] < data[f"SMA_{ma_2}"].iloc[x] and trigger!=-1:
        buy_signals.append(float("nan"))
        sell_signals.append(data["Close"].iloc[x])
        trigger = -1
    else:
        buy_signals.append(float("nan"))
        sell_signals.append(float("nan"))

data["Buy signals"] = buy_signals
data["Sell signals"] = sell_signals

plt.style.use("dark_background")
plt.plot(data["Close"], label="Share Price", color="lightgray", alpha=0.5)
plt.plot(data[f"SMA_{ma_1}"], label="30 days moving average", color="orange", linestyle="--")
plt.plot(data[f"SMA_{ma_2}"], label="100 days moving average", color="blue", linestyle="--")
plt.scatter(data.index, data["Buy signals"], label="Buy signals", marker="^", color="lime")
plt.scatter(data.index, data["Sell signals"], label="Sell signals", marker="v", color="red")
plt.legend(loc="upper left")
plt.show()
