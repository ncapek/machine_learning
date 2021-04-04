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

def get_modified_xor():
    X = np.zeros((200, 2))
    X[:50] = np.random.random((50, 2)) / 2 + 0.5
    X[50:100] = np.random.random((50, 2)) / 2
    X[100:150] = np.random.random((50, 2)) / 2 + np.array([0, 0.5])
    X[150:] = np.random.random((50, 2)) / 2 + np.array([0.5, 0])
    Y = np.array([0]*100 + [1]*100)
    return X, Y

def get_donut():
    N = 200
    R_inner = 5
    R_outer = 10

    R1 = np.random.randn(int(N/2)) + R_inner
    theta = 2*np.pi*np.random.random(int(N/2))
    X_inner = np.concatenate([[R1*np.cos(theta)], [R1*np.sin(theta)]]).T

    R2 = np.random.randn(int(N/2)) + R_outer
    theta = 2 * np.pi*np.random.random(int(N/2))
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

    X = np.concatenate([X_inner, X_outer])
    Y = np.array([0]*int(N/2) + [1]*int(N/2))

    return X, Y