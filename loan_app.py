from loan_analyst import Loan
import sys
from time import sleep


def display_menu():
    print("""
    
    -------------------------------------------------------------------------------
    |                                    Menu                                     |
    -------------------------------------------------------------------------------
    |                       1. Start new loan                                     |
    |                       2. Edit loan parameters                               |
    |                       3. Show Payment                                       |
    |                       4. Show Amortization Table                            |
    |                       5. Show Loan Summary                                  |
    |                       6. Plot Balances                                      |
    |                       7. Show size of payment to payoff in specific time    |
    |                       8. Show effect of adding amount to each payment       |
    |                       9. Compare two loans                                  |
    |                       10. Quit                                              |
    ------------------------------------------------------------------------------

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
    pv = float(input("Enter amount borrowed: "))
    amortization_type = get_amortization_type()
    loan = Loan(rate, term, pv, amortization_type)
    print("Loan initialized")
    sleep(0.75)
    return loan


def pmt(loan):
    if loan.amortization_type == "French":
        print(f" The French payment is {loan.pmt_str}")
    elif loan.amortization_type == "Italian":
        # Calcola la rata per l'ammortamento italiano
        italian_payment = loan.table['Payment'].iloc[0]
        print(f" The Italian payment is €{italian_payment:,.2f}")


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
    print(f"Monthly extra: €{result[0]:,.2f} \tTotal Payment: €{result[1]:,.2f}")

def edit_loan(loan):
    new_rate = float(input("Enter new interest rate: "))
    new_term = int(input("Enter new loan term: "))
    new_loan_amount = float(input("Enter new amount borrowed: "))
    new_amortization_type = get_amortization_type()  # Aggiungi la modifica del tipo di ammortamento
    loan.edit_loan(new_rate, new_term, new_loan_amount, new_amortization_type)
    print("Loan parameters updated.")
    sleep(1)

def compare_loans(loan1, loan2):
    # Chiamiamo il metodo `compare_loans` della classe `Loan` per confrontare i due prestiti
    Loan.compare_loans(loan1, loan2)
    sleep(2)


action = {'1': new_loan, '2': edit_loan, '3': pmt, '4': amort,
          '5': summary, '6': plot, '7': pay_early, '8':pay_faster, '9': compare_loans}


# program flow
def main():
    active_loan1 = None
    active_loan2 = None

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
            active_loan1 = loan  # Set the newly created loan as the active loan

        elif choice == '2':
            if active_loan1 and active_loan2:
                print("Active loans:")
                print("1. Active Loan 1")
                print("2. Active Loan 2")
                loan_choice = input("Select the active loan (1 or 2): ")
                if loan_choice == '1':
                    loan = active_loan1
                elif loan_choice == '2':
                    loan = active_loan2
                else:
                    print("Invalid choice. Please select 1 or 2.")
                    continue
            elif active_loan1:
                loan = active_loan1
            elif active_loan2:
                loan = active_loan2
            else:
                print("No active loans. Set up a new loan first.")
                sleep(2)
                continue

            edit_loan(loan)  # Chiamata alla funzione per modificare il prestito


        elif choice in '345678':
            if active_loan1 and active_loan2:
                print("Active loans:")
                print("1. Active Loan 1")
                print("2. Active Loan 2")
                loan_choice = input("Select the active loan (1 or 2): ")
                if loan_choice == '1':
                    loan = active_loan1
                elif loan_choice == '2':
                    loan = active_loan2
                else:
                    print("Invalid choice. Please select 1 or 2.")
                    continue
            elif active_loan1:
                loan = active_loan1
            elif active_loan2:
                loan = active_loan2
            else:
                print("No active loans. Set up a new loan first.")
                sleep(2)
                continue

            try:
                action[choice](loan)
                sleep(2)
            except NameError:
                print("No Loan setup")
                print("Set up a new loan")
                sleep(2)

        elif choice == '9':
            if active_loan1 is not None:
                rate = float(input("Enter interest rate for the second loan: "))
                term = int(input("Enter loan term for the second loan: "))
                pv = float(input("Enter amount borrowed for the second loan: "))
                amortization_type = get_amortization_type()
                loan2 = Loan(rate, term, pv, amortization_type)
                compare_loans(active_loan1, loan2)
                active_loan2 = loan2  # Set the second loan as the active loan
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
