import torch, torch.nn as nn

class GrowingModel(nn.Module):
    def __init__(self, num_classes: int, num_channels: int = 3, size: int = 28, name: str = None):
        super(GrowingModel, self).__init__()
        self.name = name if name is not None else "GrowingModel"
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

        # This reduces our convolutional features to a single vector of a fixed size. This is how we get multitask learning with the same model.
        self.fc = nn.Sequential(
            nn.Linear((size // 8) * (size // 8) * 128, 256),
            # nn.Dropout(p=0.2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64)
            # nn.BatchNorm1d(64),       Not sure if I want this here
        )

        # This acts like a multinomial logistic regression i.e. it will be our deterministic final layer
        self.classifier = nn.Sequential(
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)
        )

        self.proportions = [0.2, 0.4, 0.6, 0.8, 1.0]

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    
    def update(self):
        # based on the proportions, zero out remaining weights/biases in ALL layers to grow the model over time.
        # when update is called, the model will grow to the next proportion in self.proportions. If the model is already at 1.0, it will not grow any further.
        for module in self.modules():
            ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = self.classifier(x)
        return x