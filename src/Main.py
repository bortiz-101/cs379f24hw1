import numpy as np
import pandas as pd
import os
dirname = os.path.dirname(__file__)
datadir = os.path.join(dirname, 'data')

train_data = pd.read_csv(datadir + "/train.csv")
print(train_data.head())