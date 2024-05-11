
import pandas as pd
from lightning.pytorch import seed_everything
from torch.utils.data import DataLoader, TensorDataset

from src.moduls.config import get_default_config
from src.moduls.custom_logging import set_logging, Logger
from src.moduls.model import TransfEncModel
from src.moduls.trainer import get_trainer
from src.test.data import df2torch

import os

set_logging()
seed_everything(42, workers=True)

def get_data():
    data_path = '../../data_input/test_preproc.csv'
    return pd.read_csv(data_path)

if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    cfg = get_default_config()

    df = get_data()
    X, Y = df2torch(df[cfg.feature_label], df[cfg.target_label])

    test_dataset = TensorDataset(X, Y)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=2, persistent_workers=True)
    
    model = TransfEncModel.load_from_checkpoint(cfg.ckpt_path)

    logger = Logger(model, cfg.model_version)

    trainer = get_trainer(cfg, logger=logger)

    trainer.test(model=model, dataloaders=test_loader)


