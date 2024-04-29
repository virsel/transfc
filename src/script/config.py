from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    vocab_size: int
    n_embd: int
    n_hidden: int
    n_consecutive: int
    batch_size: int
    context_length: int
    n_workers: int
    # /path/to/save/checkpoints
    ckpt_path: [str, None] = None
    checkpoint_dir: Path = Path("../output/checkpoints/").absolute()


def get_default_config() -> Config:
    cfg =  Config(
        n_workers=12,
        vocab_size=5001,
        n_embd=16,
        n_hidden=64,
        n_consecutive=4,
        batch_size=32,
        context_length=32,
    )

    # Get the latest checkpoint file path
    latest_ckpt_file = sorted(cfg.checkpoint_dir.glob('*.ckpt'), key=lambda x: x.stat().st_mtime, reverse=True)[0]
    cfg.ckpt_path = latest_ckpt_file
    return cfg
