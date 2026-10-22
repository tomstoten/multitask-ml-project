import copy

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

    def to_compact(self, input_features: int) -> "ClassifierHeads":
        """Return a copy with each classifier restricted to active backbone features."""
        compact = copy.deepcopy(self)
        for classifier in compact.heads.values():
            layer = classifier.classifier
            compact_layer = nn.Linear(
                input_features,
                layer.out_features,
                bias=layer.bias is not None,
                device=layer.weight.device,
                dtype=layer.weight.dtype,
            )
            with torch.no_grad():
                compact_layer.weight.copy_(layer.weight[:, :input_features])
                if layer.bias is not None:
                    compact_layer.bias.copy_(layer.bias)
            classifier.classifier = compact_layer
        return compact