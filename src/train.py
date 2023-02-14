from data import read_saved_model_or_init, read_dataset_from_csv_file, save_model, Dataset
from predict import batch_predict, model
import viz
def mse(dataset:Dataset, predictions):
    if len(predictions) != dataset.n_rows: 
        raise Exception("len(predictions) != dataset.n_rows")
    sqerr = 0
    Y = dataset.get_col(1) # real data
    for i in range(dataset.n_rows):
        sqerr += (Y[i] - predictions[i])**2
    print(dataset.n_rows)

    mean = 1 / dataset.n_rows * sqerr
    return mean

def train(dataset: Dataset, learning_rate=0.0001, n_iters=2):

    tmp = model

    X = dataset.get_col(0)
    Y = dataset.get_col(1)

    for _ in range(n_iters):
        
        predictions = batch_predict(dataset.get_col(0),tmp)
        s0 = 0
        for i in range(dataset.n_rows):
            s0 += (predictions[i] - Y[i])
        s0 *= 1 / dataset.n_rows
        s0 *= learning_rate

        s1 = 0
        for i in range(dataset.n_rows):
            s1 += (predictions[i] - Y[i]) * X[i]
        s1 *= 1 / dataset.n_rows
        s1 *= learning_rate

        tmp = (tmp[0] + s0,tmp[1] + s1)

    save_model(tmp)

viz.plot_model(model)

datasets = read_dataset_from_csv_file().split()
train_dataset = datasets[0]
test_dataset = datasets[1]


predictions = batch_predict(train_dataset.get_col(0))
err = mse(train_dataset, predictions)
print(err)

viz.plot_dataset(train_dataset)

predictions = batch_predict(test_dataset.get_col(0))
err = mse(test_dataset, predictions)
print(err)

viz.plot_dataset(test_dataset)

## train

train(train_dataset, learning_rate=0.001,n_iters=1000)




viz.plt.show()




# save_model((45,45))

# print(prediction)
# print(predictions)

# from train import mse


# print(mse(dataset.))