import torch
from sklearn.model_selection import train_test_split

def df2torch(x_series, y_series):
    X = x_series.apply(lambda x: [int(v) for v in x.split()])
    X = torch.tensor(X)
    Y = torch.tensor(y_series)
    return X, Y