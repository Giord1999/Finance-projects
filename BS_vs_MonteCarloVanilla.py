import numpy as np
import scipy.stats as stats

# Funzione per calcolare il prezzo di un'opzione europea utilizzando la formula di Black-Scholes
def black_scholes(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return price

# Funzione per calcolare le greche di un'opzione europea
def calculate_greeks(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        delta = stats.norm.cdf(d1)
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = -(S * stats.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * stats.norm.cdf(d2)
    elif option_type == "put":
        delta = -stats.norm.cdf(-d1)
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = -(S * stats.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * stats.norm.cdf(-d2)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return delta, gamma, theta

# Parametri per l'opzione
S0 = 100  # Prezzo attuale del sottostante
K = 100   # Prezzo di esercizio
T = 1.0   # Scadenza in anni
r = 0.04  # Tasso di interesse senza rischio
sigma = 0.2  # Volatilità

option_type = input(str("Inserisci il tipo di opzione (call o put): "))  # Tipo di opzione (call o put)

# Calcolo del prezzo e delle greche con la formula di Black-Scholes
price_bs = black_scholes(S0, K, T, r, sigma, option_type)
delta_bs, gamma_bs, theta_bs = calculate_greeks(S0, K, T, r, sigma, option_type)

print(f"Prezzo dell'opzione (Black-Scholes): {price_bs}")
print(f"Delta (Black-Scholes): {delta_bs}")
print(f"Gamma (Black-Scholes): {gamma_bs}")
print(f"Theta (Black-Scholes): {theta_bs}")

# Simulazioni Monte Carlo con metodo delle differenze finite per calcolare le greche e il prezzo
np.random.seed(0)  # Per rendere i risultati riproducibili

num_simulations = [1000, 10000, 100000, 1000000]  # Numero di simulazioni
h = 0.01  # Piccola variazione nei parametri

for num_sim in num_simulations:
    # Variante in S0
    S_up = S0 + h
    S_down = S0 - h

    # Calcolo delle opzioni con S al variare di S0
    option_prices_up = np.zeros(num_sim)
    option_prices_down = np.zeros(num_sim)

    for i in range(num_sim):
        z = np.random.standard_normal()
        ST_up = S_up * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
        ST_down = S_down * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)

        if option_type == "call":
            payoff_up = max(ST_up - K, 0)
            payoff_down = max(ST_down - K, 0)
        elif option_type == "put":
            payoff_up = max(K - ST_up, 0)
            payoff_down = max(K - ST_down, 0)

        option_prices_up[i] = np.exp(-r * T) * payoff_up
        option_prices_down[i] = np.exp(-r * T) * payoff_down

    # Calcolo delle greche con il metodo delle differenze finite
    delta_mc = (np.mean(option_prices_up) - np.mean(option_prices_down)) / (2 * h)
    gamma_mc = (np.mean(option_prices_up) - 2 * price_bs + np.mean(option_prices_down)) / (h**2)
    theta_mc = (np.mean(option_prices_down) - 2 * price_bs + np.mean(option_prices_up)) / (2 * T * h)

    # Calcolo del prezzo dell'opzione con Monte Carlo
    option_price_mc = np.mean(option_prices_up)

    # Calcolo degli errori di approssimazione
    error_mc = abs(price_bs - option_price_mc)



    print(f"\nNumero di simulazioni: {num_sim}")
    print(f"Prezzo dell'opzione (Monte Carlo): {option_price_mc}")
    print(f"Errore di approssimazione: {error_mc}")
    print(f"Delta (Monte Carlo): {delta_mc}")
    print(f"Gamma (Monte Carlo): {gamma_mc}")
    print(f"Theta (Monte Carlo): {theta_mc}")