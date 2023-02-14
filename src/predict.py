from data import read_saved_model_or_init, read_dataset_from_csv_file, save_model, model


# input is float
def predict(input, model=model):

    output = model[1] * input + model[0] # ax + b

    return output

# inputs is array here
def batch_predict(inputs, model=model):
    return [model[1] * i + model[0] for i in inputs]

# dataset = read_dataset_from_csv_file()

# prediction = predict(dataset.data[0][0])
# predictions = batch_predict(dataset.get_col(0))

# save_model((45,45))

# print(prediction)
# print(predictions)

# from train import mse


# print(mse(dataset.))