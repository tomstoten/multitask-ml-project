from functools import partial
from . import dataloader

class Dataset():
    BLOOD = ("Blood", dataloader.load_blood_data, 8, 3)
    BREAST = ("Breast", dataloader.load_breast_data, 2, 1)
    DERMA = ("Derma", dataloader.load_derma_data, 7, 3)
    PATH = ("Path", dataloader.load_path_data, 9, 3)
    TISSUE = ("Tissue", dataloader.load_tissue_data, 8, 1)
    
    def __init__(self, name: str, load_data_func, num_classes: int, num_channels: int):
        self.name = name
        self.load_data_func = load_data_func
        self.num_classes = num_classes
        self.num_channels = num_channels

    def get_load_data_func(self, path: str | None = None, size: int | None = None):
        return partial(self.load_data_func, path=path, size=size)