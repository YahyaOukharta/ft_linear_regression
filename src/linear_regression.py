import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cache import theta_cache


class LinearRegression:
    SAVED_MODEL_NAME = "ft_model"
    LEARNING_RATE = 0.01
    N_ITERATIONS = 100

    def __init__(self, training_data_filename):
        self.training_data_filename = training_data_filename
        self.load_model()  # initializes self.theta0 and self.theta1

    def predict(self, x):
        return self.theta0 + self.theta1 * x

    def predict_with_standarized(self, x, std_theta0, std_theta1):
        return std_theta0 + std_theta1 * x

    def get_cost(self):
        data = self.load_training_data()
        if data is None:
            return
        x, y = data
        predictions = self.predict(x)
        errors = predictions - y

        # cost function : mean squared error
        cost = np.sum(errors**2) / len(x)

        return cost

    def train(self):
        data = self.load_training_data()
        if data is None:
            return
        x, y = data
        standarized_x = self.standarize_array(x)
        m = len(x)

        # saving these for visualization
        self.previous_costs = []
        self.previous_thetas = []

        # standarized thetas either from cache or 0,0
        std_theta0, std_theta1 = theta_cache.get((self.theta0, self.theta1))

        for i in range(self.N_ITERATIONS):
            # current predictions
            predictions = self.predict_with_standarized(
                standarized_x, std_theta0, std_theta1
            )
            # error
            errors = predictions - y
            # cost function : mean squared error
            cost = np.sum(errors**2) / m
            self.previous_costs.append(cost)

            # save thetas for visualizations
            self.previous_thetas.append(
                self.destandarize_theta(x, std_theta0, std_theta1)
            )

            # update theta0 and theta1 based on equations in subject
            tmp0 = self.LEARNING_RATE * np.sum(errors) / m
            tmp1 = self.LEARNING_RATE * np.sum(errors * standarized_x) / m

            std_theta0 -= tmp0
            std_theta1 -= tmp1

        self.theta0, self.theta1 = self.destandarize_theta(x, std_theta0, std_theta1)
        self.save_model()

        print(f"Trained for {self.N_ITERATIONS} iterations, final cost is {cost}")

    def visualize_model(self):
        data = self.load_training_data()
        if data is None:
            return
        x, y = data
        plt.title("Plot of data and line")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.scatter(x, y, color="red")
        xl = np.linspace(0, 250000, 50)
        plt.plot(xl, self.predict(xl))
        plt.show()
        plt.close()

    def visualize_cost_function(self):
        plt.title("Cost Function J")
        plt.xlabel("Iters")
        plt.ylabel("Cost")
        plt.plot(self.previous_costs)
        plt.show()
        plt.close()

    def visualize_animation(self):
        data = self.load_training_data()
        if data is None:
            return
        x, y = data

        fig = plt.figure()
        ax = plt.axes()
        (line,) = ax.plot([], [], lw=2)

        plt.title("Sale Price vs KM")
        plt.xlabel("KM")
        plt.ylabel("Sale Price ($)")
        plt.scatter(x, y, color="red")

        def init():
            line.set_data([], [])
            return line

        def frame(i):
            x = np.linspace(0, 300000, 50)
            y = self.previous_thetas[i][1] * x + self.previous_thetas[i][0]
            line.set_data(x, y)
            return line

        anim = animation.FuncAnimation(
            fig, frame, init_func=init, frames=self.N_ITERATIONS, interval=0
        )
        plt.show()
        plt.close()

    def load_training_data(self):
        try:
            csv_file = open(self.training_data_filename)
            rows = csv.reader(csv_file)
            next(rows)  # skip header

            x, y = [], []
            for row in rows:
                x.append(float(row[0]))
                y.append(float(row[1]))
            return np.array(x), np.array(y)

        except Exception:
            print("Training data csv file not found or invalid format")
            return None

    def standarize_array(self, x):
        return (x - np.mean(x)) / np.std(x)

    def destandarize_theta(self, x, std_theta0, std_theta1):
        non_std_theta = (
            std_theta0 - (std_theta1 * np.mean(x) / np.std(x)),
            std_theta1 / np.std(x),
        )
        theta_cache.set(
            non_std_theta,
            (std_theta0, std_theta1),
        )
        return non_std_theta

    def load_model(self):
        try:
            f = open(self.SAVED_MODEL_NAME, "r")
            self.theta0, self.theta1 = [float(theta) for theta in f.read().split(",")]
            print(f"Loaded model with theta0={self.theta0} and theta1={self.theta1}")
        except (FileNotFoundError, ValueError):
            print(
                "Model not found or has invalid format, initializing model with (0, 0)"
            )
            self.theta0, self.theta1 = 0, 0
            self.save_model()

    def save_model(self):
        with open(self.SAVED_MODEL_NAME, "w+") as f:
            f.write(f"{self.theta0},{self.theta1}")


linear_regression = LinearRegression("data.csv")
