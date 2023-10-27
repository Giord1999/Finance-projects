import datetime as dt
from dateutils import month_start, relativedelta
import matplotlib.pyplot as plt
import numpy_financial as npf
import pandas as pd

#TODO allow the user to save the results in an Excel file

"""
Autore del codice da cui ho preso ispirazione: @mmcarthy (link alla repository GitHub: https://github.com/mjmacarty). 

In questa versione del codice ho apportato delle sostanziali modifiche allo script di base. Ho inserito due tipologie di ammortamento (Italiano e Francese),
ho consentito all'utente di confrontare due prestiti, di modificarne i parametri e di scegliere con quale dei due prestiti lavorare.

Sentitevi liberi di fare dei commenti al codice e di lasciare suggerimenti. 

Fri. Oct. 27 2023

"""

class Loan:
    #Definiamo i parametri del prestito e scriviamo una funzione che consente di vedere quale prestito è attivo
    loans = []
    def __init__(self, rate, term, loan_amount, amortization_type, start=dt.date.today().isoformat()):
        self.rate = rate / 1200
        self.periods = term * 12
        self.loan_amount = loan_amount
        self.start = dt.datetime.fromisoformat(start).replace(day=1)
        self.pmt = npf.pmt(self.rate, self.periods, -self.loan_amount)
        self.pmt_str = f"€ {self.pmt:,.2f}"
        self.amortization_type = amortization_type
        self.table = self.loan_table()
        self.active = False  # Add an attribute to track if this loan is activ
        

    def set_as_active_loan(self):
        for loan in Loan.loans:
            loan.active = False  # Set all loans to inactive
        self.active = True  # Set this loan as active

    #Definiamo la struttura del piano d'ammortamento in base ai tre metodi principali (Italiano, Francese e Tedesco)
    def loan_table(self):
        periods = [self.start + relativedelta(months=x) for x in range(self.periods)]
        #Differenziamo i due tipi di ammortamento
        if self.amortization_type == "French":
            interest = [npf.ipmt(self.rate, month, self.periods, -self.loan_amount, when="end")
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
                interest_payment = (self.loan_amount - (month) * principal_payment) * self.rate
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
        if self.amortization_type == "French":
            plt.title("French Amortization Interest and Balance")
        elif self.amortization_type == "Italian":
            plt.title("Italian amortization Interest and Balance")
        else:
            title = "Unknown Amortization"
        plt.plot(amort.Balance, label='Balance (€)')
        plt.plot(amort.Interest.cumsum(), label='Interest Paid (€)')
        plt.grid(axis='y', alpha=.5)
        plt.legend(loc=8)
        plt.show()

    #Così come è utile riassumere le informazioni principali del mutuo sulla base - anche - del tipo di ammortamento utilizzato
    def summary(self):
        print("Summary")
        print("-" * 60)
        if self.amortization_type == "French":
            print(f'Payment (French amortization): {self.pmt_str:>21}')
        elif self.amortization_type == "Italian":
            italian_payment = self.table['Payment'].iloc[0]
            print(f'Payment (Italian Amortization): €{italian_payment:,.2f}')
        print(f'{"Payoff Date:":19s} {self.table.index.date[-1]}')
        print(f'Interest Paid: €{self.table["Interest"].cumsum()[-1]:,.2f}')
        print("-" * 60)

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
    # Modifica il metodo compare_loans in modo da accettare una lista di prestiti
    @classmethod
    def compare_loans(cls, loans):
        if len(loans) < 2:
            print("Please set at least two loans for comparison.")
            return

        print("Comparison of loans:")
        print("-" * 60)

        for i, loan in enumerate(loans):
            if loan.amortization_type == "French":
                monthly_payment = loan.pmt
            elif loan.amortization_type == "Italian":
                monthly_payment = loan.table['Payment'].iloc[0]
            print(f"Loan {i + 1} - Monthly Payment: €{monthly_payment:,.2f}")

        print("-" * 60)
        min_payment_loan = min(loans, key=lambda loan: loan.pmt)
        max_payment_loan = max(loans, key=lambda loan: loan.pmt)
        print(f"Loan with the lowest monthly payment: Loan {loans.index(min_payment_loan) + 1}")
        print(f"Loan with the highest monthly payment: Loan {loans.index(max_payment_loan) + 1}")

        for i, loan in enumerate(loans):
            print(f"Loan {i + 1} - Payoff Date: {loan.table.index.date[-1]}, Interest Paid: €{loan.table['Interest'].cumsum().iloc[-1]:,.2f}")

        min_interest_loan = min(loans, key=lambda loan: loan.table['Interest'].cumsum().iloc[-1])
        max_interest_loan = max(loans, key=lambda loan: loan.table['Interest'].cumsum().iloc[-1])
        print("-" * 60)
        print(f"Loan that paid the least interest: Loan {loans.index(min_interest_loan) + 1}")
        print(f"Loan that paid the most interest: Loan {loans.index(max_interest_loan) + 1}")
