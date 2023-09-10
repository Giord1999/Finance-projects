import numpy as np 
from scipy.stats import norm, binom

S0 = 100
K = 102
i = 0.05
r = np.log(1+i)
T = 1
sigma = 0.2

def BS_call(S0, K, T, r, sigma):
    d1 = (np.log(S0/K) +(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1-sigma*np.sqrt(T)
    return S0*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)

def Binomial_Call(S0, K, T,r,sigma, n):
    Dt = T/n
    u = np.exp(sigma*np.sqrt(Dt))
    d = np.exp(-sigma*np.sqrt(Dt))
    m = np.exp(r*Dt)
    q = (m-d)/(u-d)
    Sn = np.zeros(n+1)
    Cn = np.zeros(n+1)
    Qn = np.zeros(n+1)
    for k in range(0,n+1):
        Sn[k] = u**k*d**(n-k)*S0
        Cn[k] = max(Sn[k]-K, 0)
        Qn[k] = binom.pmf(k, n, q)
    return (1/(m**n))*np.dot(Cn, Qn)

print("The BS call price is ", BS_call(S0, K, T, r, sigma))
for n in range(50, 1050, 50):
    print("The binomial call price for n =",n,"is ", Binomial_Call(S0, K, T, r, sigma, n))