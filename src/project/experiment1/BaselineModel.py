import torch, torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self, num_channels: int = 3, size: int = 28, name: str = None, device: str = 'cuda'):
        super(BaselineModel, self).__init__()
        self.name = name if name is not None else "BaselineModel"
        self.device = torch.device(device)
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

        # This reduces our convolutional features to a single vector of a fixed size. This is how we get multitask learning with the same model.
        self.fc = nn.Sequential(
            nn.Linear((size // 8) * (size // 8) * 128, 256),
            # nn.Dropout(p=0.2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64)
            # nn.BatchNorm1d(64),       Not sure if I want this here
        )

        self.init_weights()

        self.to(self.device)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def grow(self):
        return -1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x