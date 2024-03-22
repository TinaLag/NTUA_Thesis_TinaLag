from main import read_csv_data, RAW_DATA_FILENAME

data = read_csv_data(RAW_DATA_FILENAME)
print(data.describe().T)

from configuration import DATA_FILENAME
from main import read_csv_data

data = read_csv_data(DATA_FILENAME)
print(data.describe().T)

import window_generator
import neural_networks
import traditional_models

neural_networks.run()
traditional_models.run()