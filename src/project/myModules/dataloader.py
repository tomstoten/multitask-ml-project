from typing import Callable
import torch.utils.data.dataloader as dataloader
import torchvision.transforms as transforms
from medmnist import BloodMNIST, BreastMNIST, DermaMNIST, PathMNIST

SUPPORTED_SIZES = (None, 64, 128, 224)

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

def get_dataloaders(data_load_fn: Callable, batch_size, shuffle=True, num_workers=0):
    """
    Creates DataLoader instances for the given dataset loading function.

    Parameters:
    - data_load_fn: A function that returns the training, validation, and test datasets.
    - batch_size: Number of samples per batch.
    - shuffle: Whether to shuffle the data (default is True).
    - num_workers: Number of subprocesses to use for data loading (default is 0).
    """
    train_data, valid_data, test_data = data_load_fn()

    print(f"Train dataset size:\t{len(train_data)}")
    print(f"Val dataset size:\t{len(valid_data)}")
    print(f"Test dataset size:\t{len(test_data)}")

    train_loader = dataloader.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, persistent_workers=True, pin_memory=True)
    valid_loader = dataloader.DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True, pin_memory=True)
    test_loader = dataloader.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True, pin_memory=True)

    
    return train_loader, valid_loader, test_loader