from linear_regression import linear_regression
import sys

linear_regression.train()

if len(sys.argv) == 1:
    print("Available visualization options: --cost, --result, --animation")

for arg in sys.argv:
    if arg == "--cost":
        linear_regression.visualize_cost_function()
    elif arg == "--result":
        linear_regression.visualize_model()
    elif arg == "--animation":
        linear_regression.visualize_animation()
