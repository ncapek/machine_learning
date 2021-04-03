import pandas as pd
import numpy as np

def get_data(limit=None):
    '''
    Load the data and return as numpy matrix
    params limit: upper limit of datapoints in case we want less
    '''
    print("Reading and transforming data...")
    df = pd.read_csv('../data/large_files/mnist.csv')
    data = df.values
    np.random.shuffle(data)
    X = data[:, 1:] / 255.0 #data is 0..255
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y

def get_alternating_grid(size=8):
    '''
    Creating a dataset of alternating points
    params size: size of square grid
    '''
    N = size ** 2
    X = np.zeros((N, 2))
    Y = np.zeros(N)
    n = 0
    start_t = 0
    for i in range(size):
        t = start_t
        for j in range(size):
            X[n] = [i, j]
            Y[n] = t
            n += 1
            t = (t + 1) % 2
        start_t = (start_t + 1) % 2
    return X, Y