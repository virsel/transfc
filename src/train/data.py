import torch
from sklearn.model_selection import train_test_split

def get_train_test_split(x_series, y_series, test_size=0.2, random_state=42):
    X = x_series.apply(lambda x: [int(v) for v in x.split()])
    X = torch.tensor(X)
    Y = torch.tensor(y_series)
    return train_test_split(X, Y, test_size=test_size, stratify=Y, random_state=random_state)