from typing import Callable
from torch.utils.data import ConcatDataset, Dataset
import torch.utils.data.dataloader as dataloader
import torchvision.transforms as transforms
import torch
from medmnist import (
    AdrenalMNIST3D,
    BloodMNIST,
    BreastMNIST,
    ChestMNIST,
    DermaMNIST,
    FractureMNIST3D,
    NoduleMNIST3D,
    OCTMNIST,
    OrganAMNIST,
    OrganMNIST3D,
    PathMNIST,
    PneumoniaMNIST,
    RetinaMNIST,
    SynapseMNIST3D,
    TissueMNIST,
    VesselMNIST3D,
)

SUPPORTED_SIZES = (None, 28, 64, 128, 224)


class TaskDataset(Dataset):
    """Attach a task name to each sample for mixed-task batches."""
    def __init__(self, dataset: Dataset, task: str):
        self.dataset = dataset
        self.task = task

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label, self.task

def _get_medmnist_size(size: int | None) -> int:
    if size not in SUPPORTED_SIZES:
        raise ValueError(f"size must be one of {SUPPORTED_SIZES}, got {size!r}")
    return 28 if size is None else size


def _augment_3d_volume(image: torch.Tensor) -> torch.Tensor:
    for spatial_axis in (1, 2, 3):
        if torch.rand(()) < 0.5:
            image = torch.flip(image, dims=(spatial_axis,))

    for plane in ((1, 2), (1, 3), (2, 3)):
        quarter_turns = int(torch.randint(0, 4, ()).item())
        if quarter_turns:
            image = torch.rot90(image, k=quarter_turns, dims=plane)
    return image


def _get_transform(augment: bool = False, channels: int = 1, is_3d: bool = False):
    image_transforms = []
    if augment and not is_3d:
        image_transforms.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=15),
        ])
    if is_3d:
        image_transforms.append(
            transforms.Lambda(lambda image: torch.as_tensor(image, dtype=torch.float32))
        )
        if augment:
            image_transforms.append(transforms.Lambda(_augment_3d_volume))
    else:
        image_transforms.append(transforms.ToTensor())
    image_transforms.append(
        transforms.Normalize(mean=[0.5] * channels, std=[0.5] * channels)
    )
    return transforms.Compose(image_transforms)

def _load_medmnist_data(
    dataset_class: type[Dataset],
    path: str,
    size: int | None,
    channels: int = 1,
    is_3d: bool = False,
):
    size = _get_medmnist_size(size)
    if is_3d and size not in (28, 64):
        raise ValueError(f"3D MedMNIST datasets support sizes 28 and 64, got {size}")
    train_transform = _get_transform(augment=True, channels=channels, is_3d=is_3d)
    eval_transform = _get_transform(channels=channels, is_3d=is_3d)
    return tuple(
        dataset_class(
            split=split,
            transform=train_transform if split == "train" else eval_transform,
            download=True,
            root=path,
            size=size,
        )
        for split in ("train", "val", "test")
    )


def load_blood_data(path="../data", size: int | None = None):
    return _load_medmnist_data(BloodMNIST, path, size, channels=3)

def load_breast_data(path="../data", size: int | None = None):
    return _load_medmnist_data(BreastMNIST, path, size)

def load_derma_data(path="../data", size: int | None = None):
    return _load_medmnist_data(DermaMNIST, path, size, channels=3)

def load_path_data(path="../data", size: int | None = None):
    return _load_medmnist_data(PathMNIST, path, size, channels=3)

def load_tissue_data(path="../data", size: int | None = None):
    return _load_medmnist_data(TissueMNIST, path, size)


def load_adrenal_data(path="../data", size: int | None = None):
    return _load_medmnist_data(AdrenalMNIST3D, path, size, is_3d=True)


def load_chest_data(path="../data", size: int | None = None):
    return _load_medmnist_data(ChestMNIST, path, size)


def load_fracture_data(path="../data", size: int | None = None):
    return _load_medmnist_data(FractureMNIST3D, path, size, is_3d=True)


def load_nodule_data(path="../data", size: int | None = None):
    return _load_medmnist_data(NoduleMNIST3D, path, size, is_3d=True)


def load_oct_data(path="../data", size: int | None = None):
    return _load_medmnist_data(OCTMNIST, path, size)


def load_organ3d_data(path="../data", size: int | None = None):
    return _load_medmnist_data(OrganMNIST3D, path, size, is_3d=True)


def load_organa_data(path="../data", size: int | None = None):
    return _load_medmnist_data(OrganAMNIST, path, size)


def load_pneumonia_data(path="../data", size: int | None = None):
    return _load_medmnist_data(PneumoniaMNIST, path, size)


def load_retina_data(path="../data", size: int | None = None):
    return _load_medmnist_data(RetinaMNIST, path, size, channels=3)


def load_synapse_data(path="../data", size: int | None = None):
    return _load_medmnist_data(SynapseMNIST3D, path, size, is_3d=True)


def load_vessel_data(path="../data", size: int | None = None):
    return _load_medmnist_data(VesselMNIST3D, path, size, is_3d=True)

def get_dataloaders(
    data_load_fn: Callable,
    batch_size,
    shuffle=True,
    num_workers=0,
    persistent_workers=False,
):
    """
    Creates DataLoader instances for the given dataset loading function.

    Parameters:
    - data_load_fn: A function that returns the training, validation, and test datasets.
    - batch_size: Number of samples per batch.
    - shuffle: Whether to shuffle the data (default is True).
    - num_workers: Number of subprocesses to use for data loading (default is 0).
    - persistent_workers: Whether workers remain alive between iterations.
    """
    train_data, valid_data, test_data = data_load_fn()

    print(f"Train dataset size:\t{len(train_data)}")
    print(f"Val dataset size:\t{len(valid_data)}")
    print(f"Test dataset size:\t{len(test_data)}")

    keep_workers = persistent_workers and num_workers > 0
    train_loader = dataloader.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, persistent_workers=keep_workers, pin_memory=True)
    valid_loader = dataloader.DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=keep_workers, pin_memory=True)
    test_loader = dataloader.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=keep_workers, pin_memory=True)


    return train_loader, valid_loader, test_loader

def get_multitask_dataloaders(
    data_load_fns: dict[str, Callable],
    batch_size,
    shuffle=True,
    num_workers=0,
    persistent_workers=False,
):
    """Build one shuffled mixed-task train loader and per-task evaluation loaders."""
    train_datasets = {}
    valid_datasets = {}
    test_datasets = {}
    for task, data_load_fn in data_load_fns.items():
        train_data, valid_data, test_data = data_load_fn()
        print(f"{task} dataset sizes: train={len(train_data)}, val={len(valid_data)}, test={len(test_data)}")
        train_datasets[task] = train_data
        valid_datasets[task] = valid_data
        test_datasets[task] = test_data

    keep_workers = persistent_workers and num_workers > 0
    train_loaders = {
        task: dataloader.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=keep_workers,
            pin_memory=True,
        )
        for task, dataset in train_datasets.items()
    }
    valid_loaders = {
        task: dataloader.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=keep_workers,
            pin_memory=True,
        )
        for task, dataset in valid_datasets.items()
    }
    test_loaders = {
        task: dataloader.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=keep_workers,
            pin_memory=True,
        )
        for task, dataset in test_datasets.items()
    }
    return train_loaders, valid_loaders, test_loaders


def get_multitask_test_loaders(
    data_load_fns: dict[str, Callable],
    batch_size,
    num_workers=0,
    persistent_workers=False,
):
    """Create deterministic test loaders without constructing train/val loaders."""
    keep_workers = persistent_workers and num_workers > 0
    test_loaders = {}
    for task, data_load_fn in data_load_fns.items():
        _, _, test_data = data_load_fn()
        print(f"{task} test dataset size:\t{len(test_data)}")
        test_loaders[task] = dataloader.DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=keep_workers,
            pin_memory=True,
        )
    return test_loaders