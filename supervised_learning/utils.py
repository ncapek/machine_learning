import pandas as pd
import numpy as np

def get_data(limit=None):
    '''
    Load the data and return as numpy matrix
    params limit: upper limit of datapoints in case we want less
    '''
    print("Reading and transforming data...")
    df = pd.read_csv('../data/mnist.csv')
    data = df.values
    np.random.shuffle(data)
    X = data[:, 1:] / 255.0 #data is 0..255
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y