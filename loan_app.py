from loan_analyst import Loan
import sys
from time import sleep

def display_menu():
    print("""
                               Menu
    ----------------------------------------------------------
    1. Start new loan
    2. Edit loan parameters
    3. Show Payment
    4. Show Amortization Table
    5. Show Loan Summary
    6. Plot Balances
    7. Show size of payment to payoff in specific time
    8. Show effect of adding amount to each payment
    9. Compare two loans
    10. Quit
    ----------------------------------------------------------

    """)


def get_amortization_type():
    while True:
        amortization_type = input("Choose amortization type (French or Italian): ").strip().title()
        if amortization_type in ["French", "Italian"]:
            return amortization_type
        else:
            print("Invalid amortization type. Please choose 'French' or 'Italian'.")

def new_loan(rate, term, pv, amortization_type):
    rate = float(input("Enter interest rate: "))
    term = int(input("Enter loan term (in years): "))
    pv = float(input("Enter loan amount: "))
    amortization_type = get_amortization_type()
    loan = Loan(rate, term, pv, amortization_type)
    print("Loan initialized")
    sleep(0.75)
    return loan



def pmt(loan):
    print(f" The payment is €{loan.pmt_str}")


def amort(loan):
    print(loan.table)


def summary(loan):
    loan.summary()


def plot(loan):
    loan.plot_balances()


def pay_faster(loan):
    amt = float(input("Enter extra monthly payment: "))
    new_term = loan.pay_early(amt)
    print(f"Paid off in {new_term} years")


def pay_early(loan):
    years_to_pay = int(input("Enter years to debt free: "))
    result = loan.retire_debt(years_to_pay)
    print(f"Monthly extra: ${result[0]:,.2f} \tTotal Payment: €{result[1]:,.2f}")

def edit_loan(loan):
    new_rate = float(input("Enter new interest rate: "))
    new_term = int(input("Enter new loan term: "))
    new_loan_amount = float(input("Enter new amount borrowed: "))
    loan.edit_loan(new_rate, new_term, new_loan_amount)
    print("Loan parameters updated.")
    sleep(1)

def compare_loans(loan1, loan2):
    Loan.compare_loans(loan1, loan2)
    sleep(2)


action = {'1': new_loan, '2': edit_loan, '3': pmt, '4': amort,
          '5': summary, '6': plot, '7': pay_early, '8':pay_faster, '9': compare_loans}


# program flow
def main():
    loan = None  # Initialize loan as None
    while True:
        display_menu()
        choice = input("Enter your selection: ")
        if choice == '1':
            rate = float(input("Enter interest rate: "))
            term = int(input("Enter loan term: "))
            pv = float(input("Enter amount borrowed: "))
            amortization_type = get_amortization_type()
            loan = Loan(rate, term, pv, amortization_type)
            print("Loan initialized")
            sleep(0.75)
        elif choice == '2':
            if loan is not None:
                edit_loan(loan)
            else:
                print("No Loan setup. Set up a new loan first.")
                sleep(2)

        elif choice in '345678':
            try:
                action[choice](loan)
                sleep(2)
            except NameError:
                print("No Loan setup")
                print("Set up a new loan")
                sleep(2)

        elif choice == '9':
            if loan is not None:
                # Create a second loan for comparison
                rate = float(input("Enter the interest rate for the second loan: "))
                term = int(input("Enter the loan term for the second loan: "))
                pv = float(input("Enter the amount borrowed for the second loan: "))
                loan2 = Loan(rate, term, pv)
                compare_loans(loan, loan2)
            else:
                print("No Loan setup. Set up a new loan first.")
                sleep(2)
        elif choice == '10':
            print("Goodbye")
            sys.exit()
        else:
            print("Please enter a valid selection")


if __name__ == "__main__":
    main()
