
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from data import get_train_test_split
from src.moduls.config import get_default_config

from src.moduls.custom_logging import set_logging, Logger
from src.moduls.model import get_model

set_logging()
import torch
import numpy as np
import random
from src.moduls.trainer import get_trainer
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
    data_path = '../../data_input/4_feature_vec.csv'
    return pd.read_csv(data_path, usecols=['id', 'feature_vec', 'label'])

if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    cfg = get_default_config()

    df = get_data()
    Xtr, Xval, Ytr, Yval = get_train_test_split(
        df["feature_vec"], df["label"], test_size=0.2, random_state=seed_value
    )

    cfg.hyper_params.context_length = Xtr.shape[1]

    train_dataset = TensorDataset(Xtr, Ytr)
    val_dataset = TensorDataset(Xval, Yval)

    train_loader = DataLoader(train_dataset, batch_size=cfg.hyper_params.batch_size, shuffle=True, num_workers=2, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.hyper_params.batch_size, shuffle=False, num_workers=2, persistent_workers=True)
    
    model = get_model(cfg.hyper_params)
    model.set_val_data_loader(val_loader)
    logger = Logger(model, cfg.model_version)

    # train with pytorch lightning
    trainer = get_trainer(cfg, logger=logger)
    trainer.fit(model, train_loader, val_loader, ckpt_path=cfg.ckpt_path)



