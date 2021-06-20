import numpy as np
import pandas as pd

def y2indicator(y):
    N = len(y)
    y = y.astype(np.int32)
    ind = np.zeros((N, 10))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

def get_normalized_data():
    print("Reading and transforming data...")
    
    df = pd.read_csv('../data/large_files/mnist.csv')
    data = df.to_numpy().astype(np.float32)
    np.random.shuffle(data)
    X = data[:, 1:]
    Y = data[:, 0]
    
    Xtrain = X[:-1000]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:]
    Ytest = Y[-1000:]
    
    # normalize
    mu = Xtrain.mean(axis=0)
    std = Xtrain.std(axis=0)
    
    # in case variance is 0
    np.place(std, std == 0, 1)
    Xtrain = (Xtrain - mu) / std
    Xtest = (Xtest - mu) / std
    
    return Xtrain, Xtest, Ytrain, Ytest

def forward(X, W, b):
    # softmax
    a = X.dot(W) + b
    expa = np.exp(a)
    y = expa / expa.sum(axis=1, keepdims=True)
    return y

def cost(p_y, t):
    tot = t * np.log(p_y)
    return -tot.sum()

def error_rate(p_y, t):
    prediction = predict(p_y)
    return np.mean(prediction != t)

def predict(p_y):
    return np.argmax(p_y, axis=1)

def gradW(t, y, X):
    return X.T.dot(t-y)

def gradb(t, y):
    return (t-y).sum(axis=0)