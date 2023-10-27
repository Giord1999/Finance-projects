from loan_analyst import Loan
import sys
from time import sleep

#TODO allow the user to save the results in an Excel file

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
    |                       9. Compare loans                                      |
    |                       10. Quit                                              |
    -------------------------------------------------------------------------------

    """)

def get_amortization_type():
    while True:
        amortization_type = input("Choose amortization type (French or Italian): ").strip().title()
        if amortization_type in ["French", "Italian"]:
            return amortization_type
        else:
            print("Invalid amortization type. Please choose 'French' or 'Italian'.")

def new_loan(loans):
    rate = float(input("Enter interest rate: "))
    term = int(input("Enter loan term (in years): "))
    pv = float(input("Enter amount borrowed: "))
    amortization_type = get_amortization_type()
    loan = Loan(rate, term, pv, amortization_type)
    loans.append(loan)
    print("Loan initialized")
    sleep(0.75)

def select_loan(loans):
    if not loans:
        print("No loans available. Please start a new loan first.")
        return None

    print("Active Loans:")
    for i, loan in enumerate(loans):
        print(f"{i + 1}. Loan with {loan.amortization_type} amortization")

    while True:
        choice = input("Select the loan for the action (1, 2, etc.): ")
        try:
            index = int(choice) - 1
            if 0 <= index < len(loans):
                return loans[index]
            else:
                print("Invalid choice. Please select a valid loan number.")
        except ValueError:
            print("Invalid input. Please enter a valid loan number.")

def pmt(loans):
    loan = select_loan(loans)
    if loan:
        if loan.amortization_type == "French":
            print(f" The French payment is {loan.pmt_str}")
        elif loan.amortization_type == "Italian":
            italian_payment = loan.table['Payment'].iloc[0]
            print(f" The Italian payment is €{italian_payment:,.2f}")

def amort(loans):
    loan = select_loan(loans)
    if loan:
        print(loan.table)

def compare_loans(loans):
    if len(loans) < 2:
        print("Please set at least two loans for comparison.")
        return

    print("Active Loans:")
    for i, loan in enumerate(loans):
        print(f"{i + 1}. Loan with {loan.amortization_type} amortization")

    selected_loans = []
    while len(selected_loans) < 2:
        choice = input("Select a loan to include in the comparison (1, 2, etc.): ")
        try:
            index = int(choice) - 1
            if 0 <= index < len(loans) and loans[index] not in selected_loans:
                selected_loans.append(loans[index])
                print(f"Loan {loans[index].amortization_type} added to comparison.")
            elif loans[index] in selected_loans:
                print("Loan is already in the comparison.")
            else:
                print("Invalid choice. Please select a valid loan number.")
        except ValueError:
            print("Invalid input. Please enter a valid loan number.")

    Loan.compare_loans(selected_loans)
    sleep(2)

def summary(loans):
    loan = select_loan(loans)
    if loan:
        loan.summary()

def plot(loans):
    loan = select_loan(loans)
    if loan:
        loan.plot_balances()

def pay_faster(loans):
    loan = select_loan(loans)
    if loan:
        amt = float(input("Enter extra monthly payment: "))
        new_term = loan.pay_early(amt)
        print(f"Paid off in {new_term} years")

def pay_early(loans):
    loan = select_loan(loans)
    if loan:
        years_to_pay = int(input("Enter years to debt free: "))
        result = loan.retire_debt(years_to_pay)
        print(f"Monthly extra: €{result[0]:,.2f} \tTotal Payment: €{result[1]:,.2f}")

def edit_loan(loans):
    loan = select_loan(loans)
    if loan:
        new_rate = float(input("Enter new interest rate: "))
        new_term = int(input("Enter new loan term: "))
        new_loan_amount = float(input("Enter new amount borrowed: "))
        new_amortization_type = get_amortization_type()
        loan.edit_loan(new_rate, new_term, new_loan_amount, new_amortization_type)
        print("Loan parameters updated.")
        sleep(1)

def main():
    loans = []

    while True:
        display_menu()
        choice = input("Enter your selection: ")
        if choice == '1':
            new_loan(loans)
        elif choice == '2':
            edit_loan(loans)
        elif choice == '3':
            pmt(loans)
        elif choice == '4':
            amort(loans)
        elif choice == '5':
            summary(loans)
        elif choice == '6':
            plot(loans)
        elif choice == '7':
            pay_early(loans)
        elif choice == '8':
            pay_faster(loans)
        elif choice == '9':
            compare_loans(loans)
        elif choice == '10':
            print("Goodbye")
            sys.exit()
        else:
            print("Please enter a valid selection")

if __name__ == "__main__":
    main()
