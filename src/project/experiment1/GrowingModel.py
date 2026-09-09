import torch, torch.nn as nn

class GrowingModel(nn.Module):
    def __init__(self, num_channels: int = 3, size: int = 28, name: str = None, device: str = 'cuda'):
        super(GrowingModel, self).__init__()
        self.name = name if name is not None else "GrowingModel"
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

        self.proportions = [0.1, 0.5, 1.0]
        self.prop_it = iter(enumerate(self.proportions))

        self.grow()

        self.to(self.device)

    def grow(self):
        try:
            index, proportion = next(self.prop_it)
        except StopIteration:
            print("All Grown Up!")
            return -1

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if index == 0:
                    module.old_size = 0
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
                    module.old_size = module.cur_size
                
                module.cur_size = int(module.weight.shape[0] * proportion)
                module.grad_mask = torch.zeros_like(module.weight, device=self.device)
                module.grad_mask[:module.cur_size] = 1
                nn.init.kaiming_normal_(module.weight[module.old_size:module.cur_size], mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(module.weight[module.cur_size:])

            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        # x = self.classifier(x)
        return x