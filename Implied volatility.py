import numpy as np 
import pandas as pd 
import yfinance as yf 
from scipy.stats import norm 
from datetime import date
import matplotlib.pyplot as plt 

#Define the Black&Scholes formula in a function
def BS_call(S0, K, T, r, sigma):
    d1 = (np.log(S0/K) +(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1-sigma*np.sqrt(T)
    return S0*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)

#Define the function to calculate the implied volatility
def imp_vol_call(S0, K, T, r, C):
    high = 10
    low = 0
    while high-low > 0.0001:
        if BS_call(S0, K, T, r, (high+low)/2)>C:
            high = (high+low)/2
        else:
            low = (high+low)/2
    return (high+low)/2

ticker = input(str("Enter your desired ticker: "))
r = float(input("Enter your desired interest rate: "))
stock = yf.Ticker(ticker)
exps = stock.options
print("The expiration dates for", ticker, "are", stock.options)
hist_data = stock.history()
S0 = hist_data["Close"].iloc[-1]
maturity = input(str("Enter your desired expiration date: "))
T = (date.fromisoformat(maturity)-date.today()).days/365
options = stock.option_chain(maturity)
calls = options.calls
calls = pd.DataFrame(calls[["strike", "lastPrice"]])
calls.rename(columns={"strike": "K", "lastPrice":"C"}, inplace=True)
calls = calls[calls["K"]<=S0*3]
calls = calls[calls["K"]>=S0*0.7]
calls["sigma"] = calls.apply(lambda row: imp_vol_call(S0, row["K"], T, r, row["C"]),axis=1)
print(calls)

#Plot the implied volatility 
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10,6))
plt.plot(calls["K"], calls["sigma"])
plt.xlabel("Strike price $K$")
plt.ylabel("Implied volatility ($\sigma$)")
plt.title("Implied volatility of" + " " + ticker + " "+ "for the maturity"+ " "+ maturity)
plt.show() 