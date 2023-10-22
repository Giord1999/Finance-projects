import datetime as dt
from dateutils import month_start, relativedelta
import matplotlib.pyplot as plt
import numpy_financial as npf
import pandas as pd

"""
Autore del codice da cui ho preso ispirazione: @mmcarthy

In questa versione del codice ho apportato delle sostanziali modifiche allo script di base. Ho inserito due tipologie di ammortamento (Italiano e Francese),
ho consentito all'utente di confrontare due prestiti, di modificarne i parametri e di scegliere con quale dei due prestiti lavorare.

Sentitevi liberi di fare dei commenti al codice e di lasciare suggerimenti. 

Sun Oct. 2 23 13:41

"""

class Loan:
    #Definiamo i parametri del prestito e scriviamo una funzione che consente di vedere quale prestito è attivo
    def __init__(self, rate, term, loan_amount, amortization_type, start=dt.date.today().isoformat()):
        self.rate = rate / 1200
        self.periods = term * 12
        self.loan_amount = loan_amount
        self.start = dt.datetime.fromisoformat(start).replace(day=1)
        self.pmt = npf.pmt(self.rate, self.periods, -self.loan_amount)
        self.pmt_str = f"€ {self.pmt:,.2f}"
        self.amortization_type = amortization_type
        self.table = self.loan_table()
        
    def set_as_active_loan(self):
        active_loan = self

    #Definiamo la struttura del piano d'ammortamento in base ai due metodi principali (Italiano o Francese)
    def loan_table(self):
        periods = [self.start + relativedelta(months=x) for x in range(self.periods)]
        #Differenziamo i due tipi di ammortamento
        if self.amortization_type == "French":
            interest = [npf.ipmt(self.rate, month, self.periods, -self.loan_amount)
                        for month in range(1, self.periods + 1)]
            principal = [npf.ppmt(self.rate, month, self.periods, -self.loan_amount)
                         for month in range(1, self.periods + 1)]
            table = pd.DataFrame({'Payment': self.pmt,
                                'Interest': interest,
                                'Principal': principal}, index=pd.to_datetime(periods))
            table['Balance'] = self.loan_amount - table['Principal'].cumsum()                      

        elif self.amortization_type == "Italian":
            interest = [self.loan_amount * self.rate]  
            principal_payment = self.loan_amount / self.periods
            principal = [principal_payment]
            payment = [interest[0] + principal[0]]

            for month in range(1, self.periods):
                interest_payment = (self.loan_amount - (month - 1) * principal_payment) * self.rate
                interest.append(interest_payment)
                principal.append(principal_payment)
                payment.append(interest_payment + principal_payment)

            principal[-1] = self.loan_amount - sum(principal[:-1])
            payment[-1] = interest[-1] + principal[-1]

            table = pd.DataFrame({'Payment': payment,
                                'Interest': interest,
                                'Principal': principal}, index=pd.to_datetime(periods))
            table['Balance'] = self.loan_amount - table['Principal'].cumsum()
        else:
            raise ValueError("Unsupported amortization type")

        return table.round(2)

    #Potrebbe essere utile pure rappresentare graficamente i risultati
    def plot_balances(self):
        amort = self.loan_table()
        plt.plot(amort.Balance, label='Balance')
        plt.plot(amort.Interest.cumsum(), label='Interest Paid')
        plt.grid(axis='y', alpha=.5)
        plt.legend(loc=8)
        plt.show()

    #Così come è utile riassumere le informazioni principali del mutuo sulla base - anche - del tipo di ammortamento utilizzato
    def summary(self):
        print("Summary")
        print("-" * 30)
        if self.amortization_type == "French":
            print(f'Payment: {self.pmt_str:>21}')
        elif self.amortization_type == "Italian":
            italian_payment = self.table['Payment'].iloc[0]
            print(f'Payment (Italian Amortization): €{italian_payment:,.2f}')
        print(f'{"Payoff Date:":19s} {self.table.index.date[-1]}')
        print(f'Interest Paid: €{self.table["Interest"].cumsum()[-1]:,.2f}')
        print("-" * 30)

    #Aggiungiamo qualche bonus: 1: vediamo cosa succede quando paghiamo di più
    def pay_early(self, extra_amt):
        return f'{round(npf.nper(self.rate, self.pmt + extra_amt, -self.loan_amount) / 12, 2)}'

    #2: Vediamo cosa succede quando impostiamo un tempo specifico per essere liberi dal mutuo
    def retire_debt(self, years_to_debt_free):
        extra_pmt = 1
        while npf.nper(self.rate, self.pmt + extra_pmt, -self.loan_amount) / 12 > years_to_debt_free:
            extra_pmt += 1
        return extra_pmt, self.pmt + extra_pmt

    #3: aggiungiamo la possibilità di modificare i parametri del prestito 
    def edit_loan(self, new_rate, new_term, new_loan_amount, new_amortization_type):
        self.rate = new_rate / 1200
        self.periods = new_term * 12
        self.loan_amount = new_loan_amount
        self.amortization_type = new_amortization_type
        self.pmt = npf.pmt(self.rate, self.periods, -self.loan_amount)
        self.pmt_str = f" € {self.pmt:,.2f}"
        self.table = self.loan_table()

    #4: aggiungiamo la possibilità di confrontare due prestiti diversi
    @staticmethod
    def compare_loans(loan1, loan2):
        loan1.set_as_active_loan()
        loan2.set_as_active_loan()
        
        print("Comparison of two loans:")
        print("-" * 30)
        
        # Calcola la rata mensile effettiva in base al tipo di ammortamento per ciascun prestito
        if loan1.amortization_type == "French":
            loan1_monthly_payment = loan1.pmt
        elif loan1.amortization_type == "Italian":
            loan1_monthly_payment = loan1.table['Payment'].iloc[0]
        
        if loan2.amortization_type == "French":
            loan2_monthly_payment = loan2.pmt
        elif loan2.amortization_type == "Italian":
            loan2_monthly_payment = loan2.table['Payment'].iloc[0]
        
        print(f"Loan 1 - Monthly Payment: €{loan1_monthly_payment:,.2f}")
        print(f"Loan 2 - Monthly Payment: €{loan2_monthly_payment:,.2f}")
        
        if loan1_monthly_payment < loan2_monthly_payment:
            print("Loan 1 has a lower monthly payment.")
        elif loan2_monthly_payment < loan1_monthly_payment:
            print("Loan 2 has a lower monthly payment.")
        else:
            print("Both loans have the same monthly payment.")
        
        # Altri confronti (payoff date, interesse pagato) rimangono invariati
        print(f"Loan 1 - Payoff Date: {loan1.table.index.date[-1]}, Interest Paid: €{loan1.table['Interest'].cumsum().iloc[-1]:,.2f}")
        print(f"Loan 2 - Payoff Date: {loan2.table.index.date[-1]}, Interest Paid: €{loan2.table['Interest'].cumsum().iloc[-1]:,.2f}")
        
        if loan1.table.index.date[-1] < loan2.table.index.date[-1]:
            print("Loan 1 has an earlier payoff date.")
        elif loan2.table.index.date[-1] < loan1.table.index.date[-1]:
            print("Loan 2 has an earlier payoff date.")
        else:
            print("Both loans have the same payoff date.")
        
        if loan1.table['Interest'].cumsum().iloc[-1] < loan2.table['Interest'].cumsum().iloc[-1]:
            print("Loan 1 has paid less interest over the life of the loan.")
        elif loan2.table['Interest'].cumsum().iloc[-1] < loan1.table['Interest'].cumsum().iloc[-1]:
            print("Loan 2 has paid less interest over the life of the loan.")
        else:
            print("Both loans have paid the same amount of interest.")
