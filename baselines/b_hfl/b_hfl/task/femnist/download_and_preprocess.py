"""Download and preprocess the femnist dataset."""

import tarfile
from pathlib import Path

import gdown

from b_hfl.utils.dataset_preparation import get_parameter_convertor


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


pathify_params = get_parameter_convertor([(str, Path)])


@pathify_params
def download_and_preprocess(download: bool, download_location: Path) -> None:
    """Download and extract the dataset.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    ## 1. print parsed config
    print("Downloading the dataset")
    if download is False:
        return

    download_femnist(dataset_dir=download_location)
