import matplotlib.pyplot as plt
import numpy as np
from data import Dataset
from data import read_saved_model_or_init, read_dataset_from_csv_file, save_model


# model should be tuple or None
def plot_dataset(dataset: Dataset, color=None):
    plt.scatter(dataset.get_col(0), dataset.get_col(1), color=color)

def plot_model(model, color=None):
        x = np.linspace(0, 400, 2)
        y = x * model[1] + model[0]
        plt.plot(x, y, label=f'y={model[1]}x+{model[0]}', color=None)
    


# dataset = read_dataset_from_csv_file()
# datasets = dataset.split()
# model = read_saved_model_or_init()



# plot_dataset(datasets[0])
# plot_dataset(datasets[1])
# plot_model(model)

# plt.show()