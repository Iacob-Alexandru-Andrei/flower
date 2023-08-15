"""Handle basic dataset creation.

In case of PyTorch it should return dataloaders for your dataset (for both the clients
and the server). If you are using a custom dataset class, this module is the place to
define it. If your dataset requires to be downloaded (and this is not done
automatically -- e.g. as it is the case for many dataset in TorchVision) and
partitioned, please include all those functions and logic in the
`dataset_preparation.py` module. You can use all those functions from functions/methods
defined here of course.
"""


import csv
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
from PIL import Image
from PIL.Image import Image as ImageType
from torch.utils.data import DataLoader, Dataset, TensorDataset
from b_hfl import strategy
from utils import lazy_wrapper

from common_types import DatasetLoaderNoTransforms, TransformType


@lazy_wrapper
def torch_tensor_to_dataset(tensor: torch.Tensor) -> Dataset:
    """Convert a torch tensor to a torch dataset."""
    return TensorDataset(tensor)


def get_load_FEMNIST_file(input_data_dir: str) -> DatasetLoaderNoTransforms:
    """Create a function that loads the FEMNIST dataset from a file.

    Specifies the data dir from which the samples are loaded.
    """
    data_dir = Path(input_data_dir)

    def load_FEMNIST_file(
        path: Path, transform: TransformType, target_transform: TransformType
    ) -> Dataset:
        """Load a FEMNIST file as a torch tensor."""
        nonlocal data_dir
        preprocessed_path: Path = path.with_suffix(".pt")
        if not preprocessed_path.exists():
            csv_path = path.with_suffix(".csv")
            if not csv_path.exists():
                raise ValueError(f"Required files do not exist, path: {csv_path}")
            else:
                with open(csv_path, mode="r") as csv_file:
                    csv_reader = csv.reader(csv_file)
                    # Ignore header
                    next(csv_reader)

                    # Extract the samples and the labels
                    partition = [
                        (sample_path, int(label_id))
                        for _, sample_path, _, label_id in csv_reader
                    ]

                    # Save for future loading
                    torch.save(partition, preprocessed_path)

        return FEMNIST(
            path=preprocessed_path,
            data_dir=data_dir,
            transform=transform,
            target_transform=target_transform,
        )

    return load_FEMNIST_file


@lazy_wrapper
def create_dataloader_FEMNIST(
    dataset: Optional[Dataset], cfg: Dict
) -> Optional[DataLoader]:
    """Create a dataloader for the FEMNIST dataset."""
    if dataset is None:
        return None
    return DataLoader(
        dataset=dataset,
        batch_size=cfg["batch_size"],
        shuffle=cfg["shuffle"],
        num_workers=cfg["num_workers"],
        drop_last=not cfg["test"],
    )


class FEMNIST(Dataset):
    """Create a PyTorch dataset from the FEMNIST dataset."""

    def __init__(
        self,
        path: Path,
        data_dir: Path,
        transform: Callable[[ImageType], torch.Tensor],
        target_transform: Callable[[int], torch.Tensor],
    ):
        self.data_dir = data_dir
        self.path = path

        self.data: Sequence[Tuple[str, int]] = torch.load(self.path)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        sample_path, label = self.data[index]

        # Convert to the full path
        full_sample_path: Path = self.data_dir / self.path.stem / sample_path

        img: ImageType = Image.open(full_sample_path).convert("L")

        return self.transform(img), self.target_transform(label)

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
            int: the length of the dataset.
        """
        return len(self.data)
