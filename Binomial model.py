import numpy as np 
import matplotlib.pyplot as plt 

S0 = 100
K = 100
i = 0.05
n = 4
u = 1.1
d = 0.9
m = 1+i
q = (m-d)/(u-d)

#American put
S = np.zeros((n+1, n+1))
S[0,0] = S0
z = 1
for t in range(1, n+1):
    for j in range(0,z):
        S[j,t] = S[j,t-1]*u
        S[j+1,t] = S[j,t-1]*d
    z +=1
PA = np.zeros_like(S)
for j in range(0, n+1):
    PA[j,n] = max(K-S[j,n], 0) 
EE = np.zeros_like(S)
z = n

for t in range(n-1, -1, -1):
    for j in range(0,z):
        if t !=0:
            P_hat = (1/m)*(q*PA[j,t+1]+(1-q)*PA[j+1, t+1])
            e_P = max(K-S[j,t], 0)
            PA[j,t] = max(P_hat, e_P)
            if e_P > P_hat:
                EE = 1
        else:
            PA[j,t] = (1/m)*(q*PA[j,t+1]+(1-q)*PA[j+1, t+1])
    z -=1

#European put
PE = np.zeros_like(S)
for j in range(0,n+1):
    PE[j,n] = max(K-S[j,n], 0)
z = n

for t in range(n-1, -1, -1):
    for j in range(0,z):
        PE[j,t] = (1/m)*(q*PE[j,t+1]+(1-q)*PE[j+1,t+1])

#Print the values of the underlying asset, the prices of the american and european puts and the value of the early exercise
print(S)
print(PA[0,0])
print(PE[0,0])
print('The value of the right to early exercise is: ', PA[0,0]-PE[0,0])