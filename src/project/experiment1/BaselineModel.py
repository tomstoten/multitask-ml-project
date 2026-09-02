import torch, torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self, name: str = None, num_channels: int = 3, num_classes: int = 7, img_size: int = 28):
        super(BaselineModel, self).__init__()
        self.name = name if name is not None else "BaselineModel"
        self.out_features = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Calculate the size of the flattened features
        flattened_size = 128 * (img_size // 8) * (img_size // 8)
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x