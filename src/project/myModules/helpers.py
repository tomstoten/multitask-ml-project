from functools import partial
from . import dataloader

DATAPATH = r"C:\Users\tomst\multitask-ml-project\src\project\data"

class Dataset():
    BLOOD = ("Blood", dataloader.load_blood_data, 8, 3)
    BREAST = ("Breast", dataloader.load_breast_data, 2, 1)
    DERMA = ("Derma", dataloader.load_derma_data, 7, 3)
    PATH = ("Path", dataloader.load_path_data, 9, 3)
    
    def __init__(self, name: str, load_data_func, num_classes: int, num_channels: int):
        self.name = name
        self.load_data_func = load_data_func
        self.num_classes = num_classes
        self.num_channels = num_channels

    def get_load_data_func(self, size: int | None = None):
        return partial(self.load_data_func, path=DATAPATH, size=size)

    @classmethod
    def get_dataset_info(cls, dataset_name: str, size: int | None = None):
        for dataset in cls:
            if dataset.name == dataset_name or dataset.name.upper() == dataset_name.upper():
                return dataset.get_load_data_func(size), dataset.num_classes
        raise ValueError(f"Dataset '{dataset_name}' not found.")