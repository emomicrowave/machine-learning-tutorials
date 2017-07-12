# Dataset source:
# http://lib.stat.cmu.edu/DASL/Stories/Forbes500CompaniesSales.html

import pandas as pd
import numpy as np

df = pd.read_csv('dataset.csv')

# create test and training data. Ratio 1:10
test_data = df.sample(8)
train_data = df.drop(df.index[test_data.index])

# create training vectors
# we'll try to predict 
