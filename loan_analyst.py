import datetime as dt
from dateutils import month_start, relativedelta
import matplotlib.pyplot as plt
import numpy_financial as npf
import pandas as pd

#TODO: allow the user to choose which of the two loans to work with

class Loan:

    def __init__(self, rate, term, loan_amount, amortization_type, start=dt.date.today().isoformat()):
        self.rate = rate / 1200
        self.periods = term * 12
        self.loan_amount = loan_amount
        self.start = dt.datetime.fromisoformat(start).replace(day=1)
        self.pmt = npf.pmt(self.rate, self.periods, -self.loan_amount)
        self.pmt_str = f"€ {self.pmt:,.2f}"
        self.amortization_type = amortization_type
        self.table = self.loan_table()

    def loan_table(self):
        periods = [self.start + relativedelta(months=x) for x in range(self.periods)]
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

    def plot_balances(self):
        amort = self.loan_table()
        plt.plot(amort.Balance, label='Balance')
        plt.plot(amort.Interest.cumsum(), label='Interest Paid')
        plt.grid(axis='y', alpha=.5)
        plt.legend(loc=8)
        plt.show()

    def summary(self):
        amort = self.table
        print("Summary")
        print("-" * 30)
        print(f'Payment: {self.pmt_str:>21}')
        print(f'{"Payoff Date:":19s} {amort.index.date[-1]}')
        print(f'Interest Paid:€{amort.Interest.cumsum()[-1]:>15,.2f}')
        print("-" * 30)

    def pay_early(self, extra_amt):
        return f'{round(npf.nper(self.rate, self.pmt + extra_amt, -self.loan_amount) / 12, 2)}'

    def retire_debt(self, years_to_debt_free):
        extra_pmt = 1
        while npf.nper(self.rate, self.pmt + extra_pmt, -self.loan_amount) / 12 > years_to_debt_free:
            extra_pmt += 1
        return extra_pmt, self.pmt + extra_pmt

    def edit_loan(self, new_rate, new_term, new_loan_amount, new_amortization_type):
        self.rate = new_rate / 1200
        self.periods = new_term * 12
        self.loan_amount = new_loan_amount
        self.amortization_type = new_amortization_type
        self.pmt = npf.pmt(self.rate, self.periods, -self.loan_amount)
        self.pmt_str = f" € {self.pmt:,.2f}"
        self.table = self.loan_table()

    @staticmethod
    def compare_loans(loan1, loan2):
        print("Comparison of two loans:")
        print(f"Loan 1 - Payment: {loan1.pmt_str}, Payoff Date: {loan1.table.index.date[-1]}, Interest Paid: €{loan1.table['Interest'].cumsum().iloc[-1]:,.2f}")
        print(f"Loan 2 - Payment: {loan2.pmt_str}, Payoff Date: {loan2.table.index.date[-1]}, Interest Paid: €{loan2.table['Interest'].cumsum().iloc[-1]:,.2f}")
        
        if loan1.pmt < loan2.pmt:
            print("Loan 1 has a lower monthly payment.")
        elif loan2.pmt < loan1.pmt:
            print("Loan 2 has a lower monthly payment.")
        else:
            print("Both loans have the same monthly payment.")

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
