import random
from csv import reader, writer

class Dataset():
    def __init__(self, features, data):
        self.features = features
        self.data = data

        self.n_rows = len(data)
        self.n_cols = len(features)
    
    def get_col(self, col_idx):
        return [row[col_idx] for row in self.data]

    def split(self):
        random.shuffle(self.data)
        test_size = self.n_rows * 20 // 100
        if not test_size: test_size = 1
        return [
            Dataset(
                self.features,
                self.data[test_size:]
            ),
            Dataset(
                self.features,
                self.data[:test_size]
            )
        ]


def read_dataset_from_csv_file(filename="data.csv"):
    with open(filename, newline='') as csvfile:
        rd = reader(csvfile, delimiter=',')
        features = []
        data = []

        for row in rd:
            if not features:
                features=(tuple(row))
            elif row:
                data.append((float(row[0]), float(row[1])))
        print("[!] dataset read from data.csv\n")
        return Dataset(features, data)

model_filename="model.csv"

def read_saved_model_or_init():
    try:
        with open(model_filename, newline='') as csvfile:
            rd = reader(csvfile, delimiter=',')
            for row in rd:
                if row:
                    model = (float(row[0]), float(row[1]))

                    print(f"[!] read {model} from model.csv\n")

                    return model
    except:
        with open(model_filename, 'w', newline='') as file:
            print("[!] model.csv created with (0,0)\n")

            wr = writer(file)
            wr.writerow((0,0))
            return (0,0,)

# model should be tuple
def save_model(model):
       with open(model_filename, 'w', newline='') as file:
            print(f"[!] model.csv saved with {model}\n")

            wr = writer(file)
            wr.writerow(model)
            refresh_model()
            return (0,0,)


# Model imported from model.csv
model = read_saved_model_or_init()
def refresh_model():
    global model
    model = read_saved_model_or_init()

