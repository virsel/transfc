from dataclasses import dataclass
from pathlib import Path


class HyperParams:
    vocab_size = 5001
    n_embd = 16
    n_hidden = 64
    batch_size = 32
    context_length = 32
    dropout = 0.1
    n_head = 1


@dataclass
class Config:
    hyper_params = HyperParams()
    n_workers = 2
    # /path/to/save/checkpoints
    ckpt_path = None
    checkpoint_dir: Path = Path("../output/checkpoints/").absolute()


def get_default_config() -> Config:
    cfg = Config()

    # Get the latest checkpoint file path
    latest_ckpt_file = sorted(
        cfg.checkpoint_dir.glob("*.ckpt"), key=lambda x: x.stat().st_mtime, reverse=True
    )
    cfg.ckpt_path = None if latest_ckpt_file == [] else latest_ckpt_file[0]
    return cfg
