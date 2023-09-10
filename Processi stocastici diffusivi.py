import numpy as np 
from numpy.random import default_rng
import matplotlib.pyplot as plt 

#Setting up the parameters
T = 2
Dt = 1/250
n = int(T*(1/Dt))
M = 90000
rng = default_rng(0)
epsilon = rng.standard_normal((n+1,M))

#Nonstandard brownian motion
mu = 0.2
sigma = 0.4
Y0 = 10
Y = np.zeros_like(epsilon)
Y[0] = Y0
for t in range(0,n):
    Y[t+1] = Y[t]+mu*Dt+sigma*np.sqrt(Dt)*epsilon[t]
plt.style.use("seaborn")
plt.rcParams["font.family"] = "serif" 
plt.figure(figsize=(10,6))
plt.title("Non standard brownian motion")
plt.xlabel("$t$")
plt.ylabel("$Y[t]$")
plt.plot(Y)
plt.show()

#Nonstandard brownian motion
mu = 0.2
sigma = 0.4
Y0 = 10
Y = np.zeros_like(epsilon)
Y[0] = Y0
for t in range(0,n):
    Y[t+1] = Y[t]+mu*Y[t]*Dt+sigma*Y[t]*np.sqrt(Dt)*epsilon[t]
    
plt.style.use("seaborn")
plt.rcParams["font.family"] = "serif" 
plt.figure(figsize=(10,6))
plt.title("Geometric brownian motion")
plt.xlabel("$t$")
plt.ylabel("$Y[t]$")
plt.plot(Y)
plt.show()

#Ornsrein-Uhlenbeck processes
alpha = 0.5
gamma = 1
Y0 = 0.2 
sigma = 0.4
Y = np.zeros_like(epsilon)
Y[0] = Y0 
for t in range(0,n):
    Y[t+1] = Y[t] + alpha*(gamma-Y[t])*Dt+sigma*np.sqrt(Dt)*epsilon[t]
plt.style.use("seaborn")
plt.rcParams["font.family"] = "serif" 
plt.figure(figsize=(10,6))
plt.title("Ornstein Uhlenbeck process")
plt.xlabel("$t$")
plt.ylabel("$Y[t]$")
plt.plot(Y)
plt.show()

#Mean reverting process
alpha = 0.5
gamma = 1
Y0 = 0.2 
sigma = 0.5
Y = np.zeros_like(epsilon)
Y[0] = Y0 
for t in range(0,n):
    Y[t+1] = Y[t] + alpha*(gamma-Y[t])*Dt+sigma*np.sqrt(Y[t])*np.sqrt(Dt)*epsilon[t]
plt.style.use("seaborn")
plt.rcParams["font.family"] = "serif" 
plt.figure(figsize=(10,6))
plt.title("Mean reverting process")
plt.xlabel("$t$")
plt.ylabel("$Y[t]$")
plt.plot(Y)
plt.show()