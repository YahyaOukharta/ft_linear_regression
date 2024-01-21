import sys
from linear_regression import linear_regression

try:
    km = float(
        input("Input a number of kilometers to predict the price of a car: ")
        if len(sys.argv) == 1
        else sys.argv[1]
    )
    prediction = linear_regression.predict(km)
    print(f"The predicted price for the km {km} is: {prediction} $")
except ValueError:
    print("Invalid input")
