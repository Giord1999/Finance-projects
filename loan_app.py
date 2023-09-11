import sys
from time import sleep
from loan_analyst import Loan

def display_menu():
    print("""
    Menu
    -----------------
    1. Create New Loan
    2. Edit Loan Parameters
    3. Compare Loans
    4. Show Payment
    5. Show Amortization Table
    6. Show Loan Summary
    7. Plot Balances
    8. Show size of payment to payoff in specific time
    9. Show effect of adding amount to each payment
    10. Quit
    """)

def create_new_loan():
    rate = float(input("Enter interest rate: "))
    term = int(input("Enter loan term: "))
    pv = float(input("Enter amount borrowed: "))
    loan = Loan(rate, term, pv)
    print("Loan initialized")
    sleep(0.75)
    return loan

def edit_loan_parameters(loan):
    rate = float(input("Enter new interest rate: "))
    term = int(input("Enter new loan term: "))
    pv = float(input("Enter new amount borrowed: "))
    loan.edit_loan_parameters(rate, term, pv)
    print("Loan parameters updated")
    sleep(0.75)

def compare_loans(loans):
    Loan.compare_loans()
    sleep(2)

def main():
    loans = []
    while True:
        display_menu()
        choice = input("Enter your selection: ")
        if choice == '1':
            loan = create_new_loan()
            loans.append(loan)

        elif choice == '2':
            if not loans:
                print("No loans available. Create a loan first.")
            else:
                compare_loans(loans)
                loan_choice = int(input("Enter the loan number to edit: ")) - 1
                if 0 <= loan_choice < len(loans):
                    edit_loan_parameters(loans[loan_choice])
                else:
                    print("Invalid loan number.")
                sleep(1)

        elif choice == '3':
            compare_loans(loans)

        elif choice in '456789':
            if not loans:
                print("No loans available. Create a loan first.")
            else:
                compare_loans(loans)
                loan_choice = int(input("Enter the loan number: ")) - 1
                if 0 <= loan_choice < len(loans):
                    action_map = {'4': loans[loan_choice].pmt,
                                  '5': loans[loan_choice].loan_table,
                                  '6': loans[loan_choice].summary,
                                  '7': loans[loan_choice].plot_balances,
                                  '8': loans[loan_choice].pay_early,
                                  '9': loans[loan_choice].retire_debt}
                    action_map[choice]()
                else:
                    print("Invalid loan number.")
                sleep(2)

        elif choice == '10':
            print("Goodbye")
            sys.exit()

        else:
            print("Please enter a valid selection")

if __name__ == "__main__":
    main()