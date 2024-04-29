
from model import get_model
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from data import get_train_test_split
from config import get_default_config
import lightning as L
from custom_logging import set_logging
set_logging()
import torch
import numpy as np
import random
from trainer import get_trainer
import os


# Set a specific seed value for reproducibility
seed_value = 42  # Choose any integer you want

# Set the seed for PyTorch (CPU and GPU if applicable)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

# Set the seed for NumPy
np.random.seed(seed_value)

# Set the seed for Python's random module
random.seed(seed_value)


def get_data():
    data_path = '../../data_input/train.csv'
    return pd.read_csv(data_path, usecols=['id', 'text', 'label'])

if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    cfg = get_default_config()

    df = get_data()
    Xtr, Xval, Ytr, Yval = get_train_test_split(
        df["text"], df["label"], context_length=cfg.context_length, test_size=0.2, random_state=seed_value
    )

    train_dataset = TensorDataset(Xtr, Ytr)
    val_dataset = TensorDataset(Xval, Yval)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2, persistent_workers=True)
    
    model = get_model(cfg)
    
    # train with pytorch lightning
    trainer = get_trainer(cfg)
    trainer.fit(model, train_loader, val_loader, ckpt_path=cfg.ckpt_path)

    # save the model
    # torch.save(model.state_dict(), "../output/model.pth")


