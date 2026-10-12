import torch
import torch.nn as nn

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
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class ClassifierHeads(nn.Module):
    """Independent classifier heads for tasks sharing one backbone."""
    def __init__(self, task_classes: dict[str, int]):
        super().__init__()
        self.heads = nn.ModuleDict({
            task: Classifier(num_classes)
            for task, num_classes in task_classes.items()
        })

    def forward(self, task: str, features: torch.Tensor) -> torch.Tensor:
        return self.heads[task](features)