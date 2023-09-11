import datetime as dt
from dateutils import month_start, relativedelta
import matplotlib.pyplot as plt
import numpy_financial as npf
import pandas as pd

class Loan:
    loans = []

    def __init__(self, rate, term, loan_amount, start=dt.date.today().isoformat()):
        self.rate = rate / 1200
        self.periods = term * 12
        self.loan_amount = loan_amount
        self.start = month_start(dt.date.fromisoformat(start) + dt.timedelta(31))
        self.pmt = npf.pmt(self.rate, self.periods, -self.loan_amount)
        self.pmt_str = f" € {self.pmt:,.2f}"
        self.table = self.loan_table()
        Loan.loans.append(self)

    def loan_table(self):
        periods = [self.start + relativedelta(months=x) for x in range(self.periods)]
        interest = [npf.ipmt(self.rate, month, self.periods, -self.loan_amount)
                    for month in range(1, self.periods + 1)]
        principal = [npf.ppmt(self.rate, month, self.periods, -self.loan_amount)
                     for month in range(1, self.periods + 1)]
        table = pd.DataFrame({'Payment': self.pmt,
                              'Interest': interest,
                              'Principal': principal}, index=pd.to_datetime(periods))
        table['Balance'] = self.loan_amount - table['Principal'].cumsum()
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
        total_payments = self.pmt * self.periods
        print("Loan Summary")
        print("-" * 30)
        print(f'Payment: {self.pmt_str:>21}')
        print(f'Payoff Date: {amort.index.date[-1]}')
        print(f'Interest Paid: {amort.Interest.cumsum()[-1]:>15,.2f}')
        print(f'Total Payments: {total_payments:>15,.2f}')
        print("-" * 30)

    def pay_early(self, extra_amt):
        return round(npf.nper(self.rate, self.pmt + extra_amt, -self.loan_amount) / 12, 2)

    def retire_debt(self, years_to_debt_free):
        extra_pmt = 1
        while npf.nper(self.rate, self.pmt + extra_pmt, -self.loan_amount) / 12 > years_to_debt_free:
            extra_pmt += 1
        return extra_pmt, self.pmt + extra_pmt

    def edit_loan_parameters(self, rate, term, loan_amount):
        self.rate = rate / 1200
        self.periods = term * 12
        self.loan_amount = loan_amount
        self.pmt = npf.pmt(self.rate, self.periods, -self.loan_amount)
        self.pmt_str = f" € {self.pmt:,.2f}"
        self.table = self.loan_table()

    @classmethod
    def compare_loans(cls):
        if len(cls.loans) < 2:
            print("You need at least two loans to compare.")
        else:
            cls.loans.sort(key=lambda loan: loan.pmt * loan.periods)
            print("Loans Comparison")
            print("-" * 30)
            for i, loan in enumerate(cls.loans):
                print(f'Loan {i + 1}')
                print(f'Payment: {loan.pmt_str}')
                print(f'Total Payments: {loan.pmt * loan.periods:,.2f}')
                print(f'Payoff Date: {loan.table.index.date[-1]}')
                print("-" * 30)