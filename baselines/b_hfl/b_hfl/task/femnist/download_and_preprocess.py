"""Download and preprocess the femnist dataset."""

import tarfile
from pathlib import Path
from typing import Callable

import gdown
import hydra
from omegaconf import DictConfig, OmegaConf

from b_hfl.utils.utils import lazy_wrapper


def download_femnist(dataset_dir: Path) -> None:
    """Download and extract the femnist dataset."""
    #  Download compressed dataset
    data_file = dataset_dir / "femnist.tar.gz"
    if not data_file.exists():
        identifier = "1-CI6-QoEmGiInV23-n_l6Yd8QGWerw8-"
        gdown.download(
            f"https://drive.google.com/uc?export=download&confirm=pbef&id={identifier}",
            str(dataset_dir / "femnist.tar.gz"),
        )

    decompressed_dataset_dir = dataset_dir / "femnist"
    # Decompress dataset
    if not decompressed_dataset_dir.exists():
        with tarfile.open(data_file, "r:gz") as tar:
            tar.extractall(decompressed_dataset_dir)

    print(f"Dataset downloaded in {dataset_dir}")


@hydra.main(config_path="conf", config_name="base", version_base=None)
def download_and_preprocess(cfg: DictConfig) -> None:
    """Download and extract the dataset.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """

    ## 1. print parsed config
    print(OmegaConf.to_yaml(cfg))
    if cfg.task.data.preparation.download is False:
        return

    download_femnist(cfg.task.data.preparation.download_location)


if __name__ == "__main__":
    download_and_preprocess()


def get_download_and_preprocess() -> Callable[[], Callable[[DictConfig], None]]:
    return download_and_preprocess
