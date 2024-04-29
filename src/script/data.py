import torch
from sklearn.model_selection import train_test_split

def get_train_test_split(x_series, y_series, context_length = 30, test_size=0.2, random_state=42):
    vocab = set(x_series.str.cat(sep=' ').split())
    vocab = sorted(list(vocab))
    # print vocab size
    stoi = {s:i+1 for i,s in enumerate(vocab)}
    stoi['.'] = 0
    print(f'Vocab size: {len(stoi)}')
    X = x_series.apply(lambda x: [stoi[v] for v in x.split()])
    X = X.apply(lambda x: [0]*(context_length-len(x)) + list(x) if len(x) < context_length  else x[:context_length])
    X = torch.tensor(X)
    torch.nn.functional.one_hot(X[0], num_classes=len(stoi)).float()
    Y = torch.tensor(y_series)
    return train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=random_state)