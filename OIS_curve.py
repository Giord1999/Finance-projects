import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from datetime import datetime
from ecbdata import ecbdata
import matplotlib.pyplot as plt
import logging
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.integrate import quad

# Configurazione logging per professionalità
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dizionario dei codici OIS
OIS_SERIES = {
    '1M': 'MMSR.B.U2._X._Z.S1ZV._Z.O._X.WR._X.FC._Z._Z.EUR._Z',
    '3M': 'MMSR.B.U2._X._Z.S1ZV._Z.O._X.WR._X.FE._Z._Z.EUR._Z',
    '6M': 'MMSR.B.U2._X._Z.S1ZV._Z.O._X.WR._X.FF._Z._Z.EUR._Z',
    '9M': 'MMSR.B.U2._X._Z.S1ZV._Z.O._X.WR._X.FG._Z._Z.EUR._Z',
    '12M': 'MMSR.B.U2._X._Z.S1ZV._Z.O._X.WR._X.FH._Z._Z.EUR._Z',
    '2Y': 'MMSR.B.U2._X._Z.S1ZV._Z.O._X.WR._X.FI._Z._Z.EUR._Z',
    '3Y': 'MMSR.B.U2._X._Z.S1ZV._Z.O._X.WR._X.FJ._Z._Z.EUR._Z',
    '5Y': 'MMSR.B.U2._X._Z.S1ZV._Z.O._X.WR._X.FK._Z._Z.EUR._Z',
    '10Y': 'MMSR.B.U2._X._Z.S1ZV._Z.O._X.WR._X.FL._Z._Z.EUR._Z',
    '>10Y': 'MMSR.B.U2._X._Z.S1ZV._Z.O._X.WR._X.FM._Z._Z.EUR._Z'
}

TENOR_TO_YEARS = {
    '1M': 1/12, '3M': 3/12, '6M': 6/12, '9M': 9/12, '12M': 1.0,
    '2Y': 2.0, '3Y': 3.0, '5Y': 5.0, '10Y': 10.0, '>10Y': 15.0
}

class OISCurve:
    """
    Classe per costruire la curva OIS usando dati ECB,
    fare bootstrapping con cubic spline, prezzare IRS e asset swap,
    e eseguire sensitivity analysis (DV01, convexity).
    Include validazione input, logging e gestione errori per massima professionalità.
    """
    def __init__(self, reference_date=None):
        self.reference_date = reference_date or datetime.today().strftime('%Y-%m-%d')
        self.tenors = list(OIS_SERIES.keys())
        self.spot_rates = None
        self.curve_spline = None
        logging.info("OISCurve inizializzata con data di riferimento: %s", self.reference_date)

    def _validate_input(self, **kwargs):
        """Validazione input privata per robustezza."""
        for key, value in kwargs.items():
            if key in ['notional', 'tenor_years'] and (not isinstance(value, (int, float)) or value <= 0):
                raise ValueError(f"{key} deve essere un numero positivo.")
            if key == 'fixed_rate' and (not isinstance(value, (int, float)) or value < 0):
                raise ValueError(f"{key} deve essere un numero non negativo.")
            if key == 'bond_price' and (not isinstance(value, (int, float)) or not (0 < value <= 1)):
                raise ValueError(f"{key} deve essere tra 0 e 1.")
            if key == 't' and (not isinstance(value, (int, float)) or value < 0):
                raise ValueError(f"{key} deve essere non negativo.")

    def fetch_data(self, lookback_days=365):
        """
        Scarica gli ultimi dati OIS da ECB utilizzando ecbdata.
        Include gestione errori e logging.
        """
        self._validate_input(lookback_days=lookback_days)
        rates = {}
        try:
            start_date = (datetime.strptime(self.reference_date, '%Y-%m-%d') - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            for tenor, code in OIS_SERIES.items():
                df = ecbdata.get_series(code, start=start_date, end=self.reference_date)
                if df.empty:
                    logging.warning("Nessun dato disponibile per %s", tenor)
                    continue
                rates[tenor] = df['OBS_VALUE'].iloc[-1] / 100  # convertiamo in decimale
            if not rates:
                raise ValueError("Nessun dato OIS scaricato.")
            self.spot_rates = pd.Series(rates)
            logging.info("Dati OIS scaricati correttamente da ECB")
        except Exception as e:
            logging.error("Errore nel fetch dei dati: %s", str(e))
            raise

    def bootstrap_curve(self):
        """Costruisce la curva OIS tramite cubic spline sui bucket disponibili."""
        if self.spot_rates is None:
            raise ValueError("Dati OIS mancanti, eseguire fetch_data() prima")
        x = np.array([TENOR_TO_YEARS[t] for t in self.tenors])
        y = np.array([self.spot_rates[t] for t in self.tenors])
        self.curve_spline = CubicSpline(x, y, bc_type='natural')
        logging.info("Curva OIS bootstrappata con cubic spline")

    def get_discount_factor(self, t):
        """Restituisce il discount factor per tempo t (in anni)."""
        self._validate_input(t=t)
        r = self.curve_spline(t)
        return np.exp(-r * t)

    def price_irs(self, notional, fixed_rate, tenor_years, freq_per_year=2):
        """
        Prezzo di un plain vanilla IRS payer (pagamento fisso contro OIS floating).
        """
        self._validate_input(notional=notional, fixed_rate=fixed_rate, tenor_years=tenor_years, freq_per_year=freq_per_year)
        dt = 1 / freq_per_year
        times = np.arange(dt, tenor_years + dt/2, dt)
        df = np.array([self.get_discount_factor(t) for t in times])
        fixed_leg = fixed_rate * dt * np.sum(df) * notional
        float_leg = (df[0] - df[-1]) * notional  # PV flussi floating con tasso OIS
        pv = float_leg - fixed_leg
        logging.info("PV IRS calcolato: %.2f", pv)
        return pv

    def price_asset_swap(self, bond_price, bond_coupon, notional, tenor_years, freq_per_year=2):
        """
        Prezzo di un asset swap semplice: bond + swap per trasformare flusso in floating
        """
        self._validate_input(bond_price=bond_price, bond_coupon=bond_coupon, notional=notional, tenor_years=tenor_years, freq_per_year=freq_per_year)
        dt = 1 / freq_per_year
        times = np.arange(dt, tenor_years + dt/2, dt)
        df = np.array([self.get_discount_factor(t) for t in times])
        fixed_leg = bond_coupon * dt * np.sum(df) * notional
        floating_leg = (self.get_discount_factor(0) - self.get_discount_factor(tenor_years)) * notional  # Correzione per accuratezza
        pv = floating_leg - fixed_leg
        logging.info("PV Asset Swap calcolato: %.2f", pv)
        return pv
    
    def calculate_dv01_curve(self, bump=0.0001):
        """
        Calcola DV01 della curva (sensibilità al 1bp parallel shift).
        Restituisce il cambiamento in PV per un bump di 1bp sui tassi spot.
        """
        if self.spot_rates is None or self.curve_spline is None:
            raise ValueError("Curva non disponibile.")
        original_rates = self.spot_rates.copy()
        bumped_rates = original_rates + bump
        # Ricostruisci spline con bumped rates
        x = np.array([TENOR_TO_YEARS[t] for t in self.tenors])
        y_bumped = np.array([bumped_rates[t] for t in self.tenors])
        spline_bumped = CubicSpline(x, y_bumped, bc_type='natural')
        # Calcola differenza nei discount factors per vari t
        t_range = np.linspace(0.1, 15, 100)
        dv01 = np.sum([np.exp(-original_rates.mean() * t) - np.exp(-spline_bumped(t) * t) for t in t_range])  # Approssimazione
        logging.info("DV01 curva calcolato: %.6f", dv01)
        return dv01

    def calculate_dv01_irs(self, notional, fixed_rate, tenor_years, freq_per_year=2, bump=0.0001):
        """
        Calcola DV01 per IRS (sensibilità al 1bp parallel shift).
        """
        pv_base = self.price_irs(notional, fixed_rate, tenor_years, freq_per_year)
        # Bump la curva
        original_spline = self.curve_spline
        x = np.array([TENOR_TO_YEARS[t] for t in self.tenors])
        y_bumped = np.array([self.spot_rates[t] + bump for t in self.tenors])
        self.curve_spline = CubicSpline(x, y_bumped, bc_type='natural')
        pv_bumped = self.price_irs(notional, fixed_rate, tenor_years, freq_per_year)
        self.curve_spline = original_spline  # Ripristina
        dv01 = pv_base - pv_bumped
        logging.info("DV01 IRS calcolato: %.2f", dv01)
        return dv01

    def calculate_convexity_irs(self, notional, fixed_rate, tenor_years, freq_per_year=2, bump=0.0001):
        """
        Calcola convexity per IRS (seconda derivata rispetto ai tassi).
        """
        pv_base = self.price_irs(notional, fixed_rate, tenor_years, freq_per_year)
        # Bump up
        original_spline = self.curve_spline
        x = np.array([TENOR_TO_YEARS[t] for t in self.tenors])
        y_up = np.array([self.spot_rates[t] + bump for t in self.tenors])
        self.curve_spline = CubicSpline(x, y_up, bc_type='natural')
        pv_up = self.price_irs(notional, fixed_rate, tenor_years, freq_per_year)
        # Bump down
        y_down = np.array([self.spot_rates[t] - bump for t in self.tenors])
        self.curve_spline = CubicSpline(x, y_down, bc_type='natural')
        pv_down = self.price_irs(notional, fixed_rate, tenor_years, freq_per_year)
        self.curve_spline = original_spline  # Ripristina
        convexity = (pv_up + pv_down - 2 * pv_base) / (bump ** 2)
        logging.info("Convexity IRS calcolata: %.4f", convexity)
        return convexity

    def plot_curve(self, num_points=100):
        """
        Plotta la curva OIS: punti spot e interpolazione cubica.
        """
        if self.curve_spline is None:
            raise ValueError("Curva non bootstrappata, eseguire bootstrap_curve() prima")
        x = np.array([TENOR_TO_YEARS[t] for t in self.tenors])
        y = np.array([self.spot_rates[t] for t in self.tenors])
        x_fine = np.linspace(x.min(), x.max(), num_points)
        y_fine = self.curve_spline(x_fine)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'ro', label='Tassi spot OIS')
        plt.plot(x_fine, y_fine, 'b-', label='Interpolazione cubica')
        plt.xlabel('Anni')
        plt.ylabel('Tasso (%)')
        plt.title('Curva OIS')
        plt.legend()
        plt.grid(True)
        plt.show()


    def _black76_call(self, F, K, T, vol, df_discount):
        """Calcola prezzo call con Black-76."""
        if T <= 0 or vol <= 0:
            return 0
        d1 = (np.log(F / K) + 0.5 * vol**2 * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        call = df_discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
        return call

    def _black76_put(self, F, K, T, vol, df_discount):
        """Calcola prezzo put con Black-76."""
        if T <= 0 or vol <= 0:
            return 0
        d1 = (np.log(F / K) + 0.5 * vol**2 * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        put = df_discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
        return put

    def _get_forward_rate(self, t_start, t_end):
        """Calcola tasso forward OIS tra t_start e t_end."""
        df_start = self.get_discount_factor(t_start)
        df_end = self.get_discount_factor(t_end)
        dt = t_end - t_start
        return (df_start / df_end - 1) / dt

    def _get_forward_swap_rate(self, swap_tenor, freq_per_year=2):
        """Calcola tasso forward swap per tenor swap_tenor."""
        dt = 1 / freq_per_year
        times = np.arange(dt, swap_tenor + dt/2, dt)
        df = np.array([self.get_discount_factor(t) for t in times])
        annuity = np.sum(df) * dt
        pv_principal = self.get_discount_factor(swap_tenor)
        return (1 - pv_principal) / annuity

    def price_cap_ois(self, strike, notional, tenor_years, freq_per_year=2, vol=0.2, num_options=None):
        """
        Prezzo di un cap OIS (serie di call su tassi floating OIS).
        Usa Black-76 per ogni periodo.
        """
        self._validate_input(strike=strike, notional=notional, tenor_years=tenor_years, freq_per_year=freq_per_year, vol=vol)
        dt = 1 / freq_per_year
        times = np.arange(dt, tenor_years + dt/2, dt)
        if num_options:
            times = times[:num_options]
        pv = 0
        for i, t in enumerate(times):
            t_start = t - dt
            t_end = t
            F = self._get_forward_rate(t_start, t_end)
            T = t_start  # Tempo all'esercizio approssimato
            df_discount = self.get_discount_factor(t)
            pv += self._black76_call(F, strike, T, vol, df_discount) * notional * dt
        logging.info("PV Cap OIS calcolato: %.2f", pv)
        return pv

    def price_floor_ois(self, strike, notional, tenor_years, freq_per_year=2, vol=0.2, num_options=None):
        """
        Prezzo di un floor OIS (serie di put su tassi floating OIS).
        Usa Black-76 per ogni periodo.
        """
        self._validate_input(strike=strike, notional=notional, tenor_years=tenor_years, freq_per_year=freq_per_year, vol=vol)
        dt = 1 / freq_per_year
        times = np.arange(dt, tenor_years + dt/2, dt)
        if num_options:
            times = times[:num_options]
        pv = 0
        for i, t in enumerate(times):
            t_start = t - dt
            t_end = t
            F = self._get_forward_rate(t_start, t_end)
            T = t_start  # Tempo all'esercizio approssimato
            df_discount = self.get_discount_factor(t)
            pv += self._black76_put(F, strike, T, vol, df_discount) * notional * dt
        logging.info("PV Floor OIS calcolato: %.2f", pv)
        return pv

    def price_swaption(self, strike, swap_tenor, option_expiry, vol, notional, is_payer=True):
        """
        Prezzo di una swaption (opzione su swap OIS).
        Usa Black-76: payer (call) o receiver (put).
        """
        self._validate_input(strike=strike, swap_tenor=swap_tenor, option_expiry=option_expiry, vol=vol, notional=notional)
        F = self._get_forward_swap_rate(swap_tenor)
        T = option_expiry
        df_discount = self.get_discount_factor(option_expiry)
        if is_payer:
            pv = self._black76_call(F, strike, T, vol, df_discount) * notional
        else:
            pv = self._black76_put(F, strike, T, vol, df_discount) * notional
        logging.info("PV Swaption calcolato: %.2f", pv)
        return pv
    

    # Hull-White Model Implementation
    def calibrate_hull_white(self, a_guess=0.1, sigma_guess=0.01):
        """
        Calibra parametri Hull-White (a, sigma) alla curva OIS.
        θ(t) è derivato per matchare la curva.
        """
        if self.spot_rates is None or self.curve_spline is None:
            raise ValueError("Curva non disponibile.")
        
        def objective(params):
            a, sigma = params
            # Calcola θ(t) per matchare r(0,t) = integral θ(s) ds - (a^2/(2a)) (1 - exp(-2a t)) + sigma^2 / (2 a^2) (1 - exp(-a t))^2
            # Approssimazione semplificata
            t_range = np.linspace(0.1, 10, 50)
            error = 0
            for t in t_range:
                r_model = self._hull_white_short_rate(a, sigma, t)
                r_market = self.curve_spline(t)
                error += (r_model - r_market)**2
            return error
        
        result = minimize(objective, [a_guess, sigma_guess], bounds=[(0.01, 1), (0.001, 0.1)])
        self.hw_a, self.hw_sigma = result.x
        logging.info("Hull-White calibrato: a=%.4f, sigma=%.4f", self.hw_a, self.hw_sigma)
        return self.hw_a, self.hw_sigma

    def _hull_white_short_rate(self, a, sigma, t):
        """Calcola r(0,t) con Hull-White."""
        integral_theta = quad(lambda s: self.curve_spline(s) + a * self.curve_spline(s) + (sigma**2 / (2 * a)) * (1 - np.exp(-a * s))**2, 0, t)[0]
        return integral_theta / t  # Approssimazione

    def simulate_hull_white(self, T, num_paths=1000, num_steps=100):
        """
        Simula percorsi del tasso a breve termine con Hull-White.
        """
        if not hasattr(self, 'hw_a') or not hasattr(self, 'hw_sigma'):
            raise ValueError("Calibrare Hull-White prima.")
        dt = T / num_steps
        r_paths = np.zeros((num_paths, num_steps + 1))
        r_paths[:, 0] = self.curve_spline(0)  # r0
        for i in range(num_steps):
            dr = (self._hull_white_theta(i * dt) - self.hw_a * r_paths[:, i]) * dt + self.hw_sigma * np.sqrt(dt) * np.random.normal(0, 1, num_paths)
            r_paths[:, i+1] = r_paths[:, i] + dr
        return r_paths

    def _hull_white_theta(self, t):
        """Calcola θ(t) per matchare la curva."""
        return self.curve_spline(t) + self.hw_a * self.curve_spline(t) + (self.hw_sigma**2 / (2 * self.hw_a)) * (1 - np.exp(-self.hw_a * t))**2

    def price_option_hull_white(self, strike, expiry, option_type='call', num_paths=10000):
        """
        Prezzo opzione europea su tasso a breve con Hull-White (es. ZCB option).
        """
        paths = self.simulate_hull_white(expiry, num_paths)
        r_T = paths[:, -1]
        df_T = np.exp(-np.sum(paths * (expiry / paths.shape[1]), axis=1))  # Approssimazione PV
        if option_type == 'call':
            payoff = np.maximum(r_T - strike, 0)
        else:
            payoff = np.maximum(strike - r_T, 0)
        pv = np.mean(payoff * df_T)
        logging.info("PV Opzione Hull-White: %.4f", pv)
        return pv

    # SABR Model Implementation
    def calibrate_sabr(self, market_vols, strikes, expiries, F, beta_guess=0.7, rho_guess=0.0, nu_guess=0.3):
        """
        Calibra parametri SABR (alpha, beta, rho, nu) ai dati di mercato.
        """
        def objective(params):
            alpha, beta, rho, nu = params
            error = 0
            for vol, K, T in zip(market_vols, strikes, expiries):
                vol_model = self._sabr_volatility(F, K, T, alpha, beta, rho, nu)
                error += (vol_model - vol)**2
            return error
        
        result = minimize(objective, [0.2, beta_guess, rho_guess, nu_guess], bounds=[(0.01, 1), (0, 1), (-0.99, 0.99), (0.01, 1)])
        self.sabr_alpha, self.sabr_beta, self.sabr_rho, self.sabr_nu = result.x
        logging.info("SABR calibrato: alpha=%.4f, beta=%.4f, rho=%.4f, nu=%.4f", self.sabr_alpha, self.sabr_beta, self.sabr_rho, self.sabr_nu)
        return result.x

    def _sabr_volatility(self, F, K, T, alpha, beta, rho, nu):
        """Calcola vol implicita SABR."""
        if F == K:
            return alpha / (F**(1 - beta))
        z = (nu / alpha) * (F * K)**((1 - beta)/2) * np.log(F / K)
        x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
        A = alpha / ((F * K)**((1 - beta)/2) * (1 + ((1 - beta)**2 / 24) * (np.log(F/K))**2 + ((1 - beta)**4 / 1920) * (np.log(F/K))**4))
        B = 1 + (((1 - beta)**2 / 24) * alpha**2 / ((F * K)**(1 - beta)) + 0.25 * rho * beta * nu * alpha / ((F * K)**((1 - beta)/2)) + (2 - 3 * rho**2) * nu**2 / 24) * T
        return A * (z / x_z) * B

    def price_swaption_sabr(self, strike, swap_tenor, option_expiry, notional, is_payer=True):
        """
        Prezzo swaption usando vol SABR in Black-76.
        """
        if not hasattr(self, 'sabr_alpha'):
            raise ValueError("Calibrare SABR prima.")
        F = self._get_forward_swap_rate(swap_tenor)
        vol = self._sabr_volatility(F, strike, option_expiry, self.sabr_alpha, self.sabr_beta, self.sabr_rho, self.sabr_nu)
        pv = self.price_swaption(strike, swap_tenor, option_expiry, vol, notional, is_payer)
        logging.info("PV Swaption SABR: %.4f", pv)
        return pv



    # CIR Model Implementation
    def calibrate_cir(self, a_guess=0.1, b_guess=0.03, sigma_guess=0.01):
        """
        Calibra parametri CIR (a, b, sigma) alla curva OIS.
        """
        if self.spot_rates is None or self.curve_spline is None:
            raise ValueError("Curva non disponibile.")
        
        def objective(params):
            a, b, sigma = params
            t_range = np.linspace(0.1, 10, 50)
            error = 0
            for t in t_range:
                r_model = self._cir_short_rate(a, b, sigma, t)
                r_market = self.curve_spline(t)
                error += (r_model - r_market)**2
            return error
        
        result = minimize(objective, [a_guess, b_guess, sigma_guess], bounds=[(0.01, 1), (0.01, 0.1), (0.001, 0.1)])
        self.cir_a, self.cir_b, self.cir_sigma = result.x
        logging.info("CIR calibrato: a=%.4f, b=%.4f, sigma=%.4f", self.cir_a, self.cir_b, self.cir_sigma)
        return result.x

    def _cir_short_rate(self, a, b, sigma, t):
        """Calcola r(0,t) con CIR."""
        integral_theta = quad(lambda s: a * b + (sigma**2 / (2 * a)) * (1 - np.exp(-a * s))**2, 0, t)[0]
        return integral_theta / t  # Approssimazione semplificata

    def simulate_cir(self, T, num_paths=1000, num_steps=100):
        """
        Simula percorsi del tasso a breve termine con CIR.
        """
        if not hasattr(self, 'cir_a') or not hasattr(self, 'cir_b') or not hasattr(self, 'cir_sigma'):
            raise ValueError("Calibrare CIR prima.")
        dt = T / num_steps
        r_paths = np.zeros((num_paths, num_steps + 1))
        r_paths[:, 0] = self.curve_spline(0)
        for i in range(num_steps):
            dr = self.cir_a * (self.cir_b - r_paths[:, i]) * dt + self.cir_sigma * np.sqrt(np.maximum(r_paths[:, i], 0)) * np.sqrt(dt) * np.random.normal(0, 1, num_paths)
            r_paths[:, i+1] = np.maximum(r_paths[:, i] + dr, 0)  # Garantisci positività
        return r_paths

    def price_option_cir(self, strike, expiry, option_type='call', num_paths=10000):
        """
        Prezzo opzione europea su tasso a breve con CIR.
        """
        paths = self.simulate_cir(expiry, num_paths)
        r_T = paths[:, -1]
        df_T = np.exp(-np.sum(paths * (expiry / paths.shape[1]), axis=1))
        if option_type == 'call':
            payoff = np.maximum(r_T - strike, 0)
        else:
            payoff = np.maximum(strike - r_T, 0)
        pv = np.mean(payoff * df_T)
        logging.info("PV Opzione CIR: %.4f", pv)
        return pv

    # Black-Karasinski Model Implementation
    def calibrate_black_karasinski(self, a_guess=0.1, sigma_guess=0.01):
        """
        Calibra parametri Black-Karasinski (a, sigma) alla curva OIS.
        θ(t) è derivato per matchare la curva.
        """
        if self.spot_rates is None or self.curve_spline is None:
            raise ValueError("Curva non disponibile.")
        
        def objective(params):
            a, sigma = params
            t_range = np.linspace(0.1, 10, 50)
            error = 0
            for t in t_range:
                r_model = self._bk_short_rate(a, sigma, t)
                r_market = self.curve_spline(t)
                error += (r_model - r_market)**2
            return error
        
        result = minimize(objective, [a_guess, sigma_guess], bounds=[(0.01, 1), (0.001, 0.1)])
        self.bk_a, self.bk_sigma = result.x
        logging.info("Black-Karasinski calibrato: a=%.4f, sigma=%.4f", self.bk_a, self.bk_sigma)
        return result.x

    def _bk_short_rate(self, a, sigma, t):
        """Calcola r(0,t) con Black-Karasinski (approssimazione log-normale)."""
        integral_theta = quad(lambda s: self.curve_spline(s) + a * np.log(self.curve_spline(s)) + (sigma**2 / (2 * a)) * (1 - np.exp(-a * s))**2, 0, t)[0]
        return np.exp(integral_theta / t)  # Trasformazione log-normale

    def simulate_black_karasinski(self, T, num_paths=1000, num_steps=100):
        """
        Simula percorsi del tasso a breve termine con Black-Karasinski.
        """
        if not hasattr(self, 'bk_a') or not hasattr(self, 'bk_sigma'):
            raise ValueError("Calibrare Black-Karasinski prima.")
        dt = T / num_steps
        ln_r_paths = np.zeros((num_paths, num_steps + 1))
        ln_r_paths[:, 0] = np.log(self.curve_spline(0))
        for i in range(num_steps):
            d_ln_r = (self._bk_theta(i * dt) - self.bk_a * ln_r_paths[:, i]) * dt + self.bk_sigma * np.sqrt(dt) * np.random.normal(0, 1, num_paths)
            ln_r_paths[:, i+1] = ln_r_paths[:, i] + d_ln_r
        return np.exp(ln_r_paths)  # Torna a r

    def _bk_theta(self, t):
        """Calcola θ(t) per Black-Karasinski."""
        return self.curve_spline(t) + self.bk_a * np.log(self.curve_spline(t)) + (self.bk_sigma**2 / (2 * self.bk_a)) * (1 - np.exp(-self.bk_a * t))**2

    def price_option_bk(self, strike, expiry, option_type='call', num_paths=10000):
        """
        Prezzo opzione europea su tasso a breve con Black-Karasinski.
        """
        paths = self.simulate_black_karasinski(expiry, num_paths)
        r_T = paths[:, -1]
        df_T = np.exp(-np.sum(paths * (expiry / paths.shape[1]), axis=1))
        if option_type == 'call':
            payoff = np.maximum(r_T - strike, 0)
        else:
            payoff = np.maximum(strike - r_T, 0)
        pv = np.mean(payoff * df_T)
        logging.info("PV Opzione Black-Karasinski: %.4f", pv)
        return pv


# --- Esempio di utilizzo con sensitivity ---
# Inizializza la curva OIS
curve = OISCurve(reference_date='2025-10-07')

# Scarica dati OIS da ECB
curve.fetch_data()

# Bootstrap della curva con cubic spline
curve.bootstrap_curve()

# Prezzo un IRS payer
pv_irs = curve.price_irs(notional=1e6, fixed_rate=0.03, tenor_years=5)
print(f"PV IRS: {pv_irs}")

# Calcola DV01 e convexity per l'IRS
dv01_irs = curve.calculate_dv01_irs(notional=1e6, fixed_rate=0.03, tenor_years=5)
print(f"DV01 IRS: {dv01_irs}")
convexity_irs = curve.calculate_convexity_irs(notional=1e6, fixed_rate=0.03, tenor_years=5)
print(f"Convexity IRS: {convexity_irs}")

# Prezzo un asset swap
pv_asset_swap = curve.price_asset_swap(bond_price=0.95, bond_coupon=0.04, notional=1e6, tenor_years=5)
print(f"PV Asset Swap: {pv_asset_swap}")

# Prezzo un cap OIS
pv_cap = curve.price_cap_ois(strike=30, notional=1e6, tenor_years=5, vol=0.2)
print(f"PV Cap OIS: {pv_cap}")

# Prezzo un floor OIS
pv_floor = curve.price_floor_ois(strike=30, notional=1e6, tenor_years=5, vol=0.2)
print(f"PV Floor OIS: {pv_floor}")

# Prezzo una swaption payer
pv_swaption = curve.price_swaption(strike=30, swap_tenor=5, option_expiry=1, vol=0.2, notional=1e6, is_payer=True)
print(f"PV Swaption Payer: {pv_swaption}")

# Calibra e usa Hull-White
a, sigma = curve.calibrate_hull_white()
print(f"Hull-White params: a={a}, sigma={sigma}")

# Simula percorsi Hull-White
paths = curve.simulate_hull_white(T=5, num_paths=100, num_steps=50)
print(f"Simulazione completata, shape percorsi: {paths.shape}")

# Prezzo un'opzione con Hull-White
pv_option_hw = curve.price_option_hull_white(strike=0.03, expiry=1, option_type='call')
print(f"PV Opzione Hull-White: {pv_option_hw}")

# Calibra SABR (esempio con dati fittizi)
market_vols = [0.2, 0.21, 0.19]  # Volatilità di mercato esempio
strikes = [0.03, 0.032, 0.028]
expiries = [1, 1, 1]
F = curve._get_forward_swap_rate(5)  # Forward rate per swap
alpha, beta, rho, nu = curve.calibrate_sabr(market_vols, strikes, expiries, F)
print(f"SABR params: alpha={alpha}, beta={beta}, rho={rho}, nu={nu}")

# Prezzo swaption con SABR
pv_swaption_sabr = curve.price_swaption_sabr(strike=30, swap_tenor=5, option_expiry=1, notional=1e6, is_payer=True)
print(f"PV Swaption SABR: {pv_swaption_sabr}")

# Plotta la curva
curve.plot_curve()

# Calibra e usa CIR
a_cir, b_cir, sigma_cir = curve.calibrate_cir()
print(f"CIR params: a={a_cir}, b={b_cir}, sigma={sigma_cir}")
pv_option_cir = curve.price_option_cir(strike=30, expiry=1, option_type='call')
print(f"PV Opzione CIR: {pv_option_cir}")

# Calibra e usa Black-Karasinski
a_bk, sigma_bk = curve.calibrate_black_karasinski()
print(f"Black-Karasinski params: a={a_bk}, sigma={sigma_bk}")
pv_option_bk = curve.price_option_bk(strike=30, expiry=1, option_type='call')
print(f"PV Opzione Black-Karasinski: {pv_option_bk}")
