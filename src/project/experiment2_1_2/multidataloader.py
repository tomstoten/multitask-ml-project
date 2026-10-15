from typing import Callable
from torch.utils.data import ConcatDataset, Dataset
import torch.utils.data.dataloader as dataloader
import torchvision.transforms as transforms
from medmnist import BloodMNIST, BreastMNIST, DermaMNIST, PathMNIST, TissueMNIST

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

def load_blood_data(path="../data", size: int | None = None):
    """
    Input:
        path: the path of data

    Output (Return values):
        train_data: The training data
        valid_data: The validation data
        test_data:  The test data
    """
    size = _get_medmnist_size(size)
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load data
    train_data = BloodMNIST(split="train", transform=data_transform, root=path, size=size)
    valid_data = BloodMNIST(split="val", transform=data_transform, root=path, size=size)
    test_data = BloodMNIST(split="test", transform=data_transform, root=path, size=size)
    return train_data, valid_data, test_data

def load_breast_data(path="../data", size: int | None = None):
    """
    Input:
        path: the path of data

    Output (Return values):
        train_data: The training data
        valid_data: The validation data
        test_data:  The test data
    """
    size = _get_medmnist_size(size)
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load data
    train_data = BreastMNIST(split="train", transform=data_transform, root=path, size=size)
    valid_data = BreastMNIST(split="val", transform=data_transform, root=path, size=size)
    test_data = BreastMNIST(split="test", transform=data_transform, root=path, size=size)
    return train_data, valid_data, test_data

def load_derma_data(path="../data", size: int | None = None):
    """
    Input:
        path: the path of data

    Output (Return values):
        train_data: The training data
        valid_data: The validation data
        test_data:  The test data
    """
    size = _get_medmnist_size(size)
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load data
    train_data = DermaMNIST(split="train", transform=data_transform, root=path, size=size)
    valid_data = DermaMNIST(split="val", transform=data_transform, root=path, size=size)
    test_data = DermaMNIST(split="test", transform=data_transform, root=path, size=size)
    return train_data, valid_data, test_data

def load_path_data(path="../data", size: int | None = None):
    """
    Input:
        path: the path of data

    Output (Return values):
        train_data: The training data
        valid_data: The validation data
        test_data:  The test data
    """
    size = _get_medmnist_size(size)
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load data
    train_data = PathMNIST(split="train", transform=data_transform, root=path, size=size)
    valid_data = PathMNIST(split="val", transform=data_transform, root=path, size=size)
    test_data = PathMNIST(split="test", transform=data_transform, root=path, size=size)
    return train_data, valid_data, test_data

def load_tissue_data(path="../data", size: int | None = None):
    """
    Input:
        path: the path of data
        size: the size of the images

    Output (Return values):
        train_data: The training data
        valid_data: The validation data
        test_data:  The test data
    """
    size = _get_medmnist_size(size)
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load data
    train_data = TissueMNIST(split="train", transform=data_transform, root=path, size=size)
    valid_data = TissueMNIST(split="val", transform=data_transform, root=path, size=size)
    test_data = TissueMNIST(split="test", transform=data_transform, root=path, size=size)
    return train_data, valid_data, test_data

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