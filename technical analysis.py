import pandas as pd 
import yfinance as yf 
import pandas_datareader as pdr
import matplotlib.pyplot as plt 
import datetime as dt 
import mplfinance as mpf

ticker = "META"
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

#Plot del relative strenght index
plt.figure(figsize=(12,8))
ax1 = plt.subplot(311)
ax1.set_title("Stock Price for {}".format(ticker), color="white")   #Titolo del grafico
ax1.plot(combined_df.index, combined_df["Close"], color = "lightgray")
ax1.grid(True, color = "#555555")
ax1.set_axisbelow(True)
ax1.set_facecolor("black")
ax1.figure.set_facecolor("#121212")
ax1.tick_params(axis="x", colors="white")
ax1.tick_params(axis="y", colors="white")

ax2 = plt.subplot(312, sharex=ax1)
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


plt.show()
#Plot del candlestick graph
mpf.plot(data=data, type="candle", style="yahoo", volume=True)