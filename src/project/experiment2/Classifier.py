import torch, torch.nn as nn

class Classifier(nn.Module):
    '''This classifier acts as a multinomial logistic regression i.e. it will be our deterministic final layer.
    Input:
        num_classes: The number of classes in the dataset.
    Output:
        x: The predicted probabilities for each class.
    '''
    def __init__(self, num_classes: int):
        super(Classifier, self).__init__()
        self.out_features = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.classifier(x)
        return x