from dataclasses import dataclass
from pathlib import Path


class HyperParams:
    def __init__(self):
        self.vocab_size = 5001
        self.n_embd = 16
        self.n_hidden = 64
        self.batch_size = 32
        self.context_length = 32
        self.dropout = 0.1
        self.n_head = 1


@dataclass
class Config:
    hyper_params = HyperParams()
    n_workers = 4
    model_version = "v1"
    # /path/to/save/checkpoints
    ckpt_path = None
    checkpoint_dir: Path = Path(f"../output/checkpoints/{model_version}").absolute()


def get_default_config() -> Config:
    cfg = Config()

    # Ensure the checkpoint directory exists, create it if not
    if not cfg.checkpoint_dir.exists():
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Get the latest checkpoint file path
    latest_ckpt_file = sorted(
        cfg.checkpoint_dir.glob("*.ckpt"), key=lambda x: x.stat().st_mtime, reverse=True
    )
    cfg.ckpt_path = None if latest_ckpt_file == [] else latest_ckpt_file[0]
    return cfg
