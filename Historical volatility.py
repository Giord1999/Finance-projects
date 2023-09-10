import numpy as np 
import pandas as pd 
import statistics as st 
import yfinance as yf 
import matplotlib.pyplot as plt
from scipy.stats import shapiro, norm, t

#Single stock
##1 Calculating historical volatility
ticker = input(str("Enter your desired ticker: "))
stock = yf.Ticker(ticker)
hist_data = stock.history(period="max")
prices = pd.DataFrame(hist_data["Close"])
prices.sort_index(ascending=False,inplace=True)
prices["LogRet"] = (np.log(prices["Close"]/prices["Close"].shift(-1)))
log_ret = prices["LogRet"]
log_ret = log_ret[:-1]
vol_day = st.stdev(log_ret)
vol_year = vol_day*np.sqrt(250)
print("Historical volatility of", ticker, "is", vol_year)

##2 Setting up parameters for distribution fitting: Normal and t
[mean_normal_fit, std_normal_fit] = norm.fit(log_ret)
[df, loc, scale] = t.fit(log_ret)
x = np.linspace(np.min(log_ret), np.max(log_ret))

##3 Checking for normality graphically: plotting the normal and the t distribution
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10,6))
plt.hist(log_ret, bins=500, density=True, color="blue")
plt.plot(x, norm.pdf(x, mean_normal_fit, std_normal_fit), color= "red", linewidth=2)
plt.plot(x, t.pdf(x, df=df, loc=loc, scale=scale), color= "yellow", linewidth=2)
plt.legend(["Normal distribution fit", "t distribution fit"])
plt.xlabel("Log return of stock price")
plt.ylabel("Frequency of Log returns")
plt.title("Log return distribution for" + " " + ticker)
plt.show()

##4 Checking for normality statistically: the Shapiro-Wilk test
Normality_test = shapiro(log_ret)
if Normality_test[1] >= 0.05:
    print(Normality_test)
    print("The sample of log returns is normally distributed")
else:
    print(Normality_test)
    print("The sample of log returns is not normally distributed")

#Multi asset analysis
##1a Define the function
def hist_vol(ticker):
    stock = yf.Ticker(ticker)
    hist_data = stock.history(period="max")
    prices = pd.DataFrame(hist_data["Close"])
    prices.sort_index(ascending=False,inplace=True)
    prices["LogRet"] = (np.log(prices["Close"]/prices["Close"].shift(-1)))
    log_ret = prices["LogRet"]
    log_ret = log_ret[:-1]
    vol_day = st.stdev(log_ret)
    vol_year = vol_day*np.sqrt(250)
    return vol_year

print("Comparison of different stocks: ")
tickers = np.array(["AAPL", "AMZN", "GOOG", "STLA", "TSLA"])
for i in range(0, len(tickers)):
    vol_year = hist_vol(tickers[i])
    print("Historical volatility of", tickers[i], "is", vol_year)