import torch
import torch.nn as nn


def _initialize_weights(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.BatchNorm1d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


class _Model2DArchitecture(nn.Module):
    spatial_dims = 2

    def __init__(
        self,
        num_channels: int = 3,
        size: int = 28,
        name: str | None = None,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.name = name or type(self).model_name
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
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear((size // 8) ** 2 * 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )
        _initialize_weights(self)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected a 2D image batch with 4 dimensions, got shape {tuple(x.shape)}")
        x = self.features(x)
        return self.fc(torch.flatten(x, start_dim=1))


class _Model3DArchitecture(nn.Module):
    spatial_dims = 3

    def __init__(
        self,
        num_channels: int = 1,
        size: int = 28,
        name: str | None = None,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.name = name or type(self).model_name
        self.device = torch.device(device)
        self.features = nn.Sequential(
            nn.Conv3d(num_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.spatial_pool = nn.AdaptiveAvgPool3d(2)
        self.fc = nn.Sequential(
            nn.Linear(128 * 2 * 2 * 2, 256),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )
        _initialize_weights(self)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected a 3D image batch with 5 dimensions, got shape {tuple(x.shape)}")
        x = self.features(x)
        x = self.spatial_pool(x)
        return self.fc(torch.flatten(x, start_dim=1))


class BaselineModel2D(_Model2DArchitecture):
    model_name = "BaselineModel"


class BaselineModel3D(_Model3DArchitecture):
    model_name = "BaselineModel"


class BaselineModel:
    def __new__(
        cls,
        num_channels: int = 3,
        size: int = 28,
        name: str | None = None,
        device: str | torch.device = "cuda",
        spatial_dims: int = 2,
    ) -> nn.Module:
        if spatial_dims == 2:
            model_type = BaselineModel2D
        elif spatial_dims == 3:
            model_type = BaselineModel3D
        else:
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
        return model_type(num_channels=num_channels, size=size, name=name, device=device)
