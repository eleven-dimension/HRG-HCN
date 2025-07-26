from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

CONFIG_DIR = (Path(__file__).resolve().parents[2] / "configs").absolute()


def test_block_init():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config")

    block = instantiate(cfg.model)
    assert block is not None
