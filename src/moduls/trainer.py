from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from .config import Config


def get_trainer(cfg: Config, logger=None):
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=cfg.checkpoint_dir,
        filename='ckpt-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode="min",
    )
    return Trainer(
        # checkpointing
        callbacks=[checkpoint_callback],
        # logging
        # distribution
        accelerator='cpu',
        strategy="ddp_spawn",
        devices=cfg.n_workers,
        # Other trainer arguments
        max_epochs=25,
        logger=logger
    )
