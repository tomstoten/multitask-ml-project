from typing import Any, Dict, Union
import json, os
import torch
from torch.utils.tensorboard import SummaryWriter
from project.myModules import dataloader, timestamp, plotting, helpers
import BaselineModel, GrowingModel

def train_model(model: torch.nn.Module, 
                train_dataloader: torch.utils.data.DataLoader, 
                valid_dataloader: torch.utils.data.DataLoader, 
                num_epochs: int = 100, 
                learning_rate: float = 0.001, 
                device: str = 'cuda',
                tag: str = "run") -> Union[list, list]:
    """
    Train the given model using the provided training and validation data loaders.

    Input:
        model: The model to be trained (BaselineModel or GrowingModel)
        train_dataloader: The training data loader
        valid_dataloader: The validation data loader
        num_epochs: Number of epochs for training
        learning_rate: Learning rate for the optimizer
        device: Device to run the training on ('cuda' or 'cpu')
        tag: Tag for the training run
    """
    model.to(device)
    print(f"Training on device: {device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    train_losses = []
    valid_losses = []

    tensorboard_writer = SummaryWriter(log_dir=f'./runs/{tag}')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_dataloader:
            inputs: torch.Tensor = inputs.to(device)
            labels: torch.Tensor = labels.to(device)
            # make labels one-hot encoded for CrossEntropyLoss
            labels = torch.nn.functional.one_hot(labels, num_classes=model.out_features).float().squeeze()

            optimizer.zero_grad()
            outputs: torch.Tensor = model(inputs)
            
            loss: torch.Tensor = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_dataloader.dataset)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, labels in valid_dataloader:
                inputs: torch.Tensor = inputs.to(device)
                labels: torch.Tensor = labels.to(device)
                # make labels one-hot encoded for CrossEntropyLoss
                labels = torch.nn.functional.one_hot(labels, num_classes=model.out_features).float().squeeze()
                
                outputs: torch.Tensor = model(inputs)
                loss: torch.Tensor = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)

        valid_loss /= len(valid_dataloader.dataset)

        tensorboard_writer.add_scalar('Loss/Train', train_loss, epoch)
        tensorboard_writer.add_scalar('Loss/Validation', valid_loss, epoch)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

    tensorboard_writer.close()
    print("Training complete.")
    return train_losses, valid_losses

def evaluate_model(model: torch.nn.Module,
                   test_dataloader: torch.utils.data.DataLoader,
                   device: str = 'cuda') -> Dict[str, Any]:
    """
    Evaluate the given model using the provided test data loader.

    Input:
        model: The model to be evaluated (BaselineModel or GrowingModel)
        test_dataloader: The test data loader
        device: Device to run the evaluation on ('cuda' or 'cpu')

    Output:
        metrics: Overall and classwise evaluation metrics
    """
    model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs: torch.Tensor = inputs.to(device)
            labels: torch.Tensor = labels.to(device).view(-1).long()
            outputs: torch.Tensor = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = probabilities.argmax(dim=1)

            all_labels.append(labels.cpu())
            all_predictions.append(predictions.cpu())
            all_probabilities.append(probabilities.cpu())

    labels = torch.cat(all_labels)
    predictions = torch.cat(all_predictions)
    probabilities = torch.cat(all_probabilities)
    num_classes = probabilities.shape[1]
    classwise_metrics = {}
    auc_values = []

    for class_index in range(num_classes):
        actual = labels == class_index
        predicted = predictions == class_index
        true_positive = (actual & predicted).sum().item()
        false_positive = (~actual & predicted).sum().item()
        false_negative = (actual & ~predicted).sum().item()
        true_negative = (~actual & ~predicted).sum().item()

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0

        positive_count = actual.sum().item()
        negative_count = (~actual).sum().item()
        if positive_count and negative_count:
            sorted_indices = torch.argsort(probabilities[:, class_index])
            sorted_actual = actual[sorted_indices].to(torch.float64)
            ranks = torch.arange(1, len(labels) + 1, dtype=torch.float64)
            auc = ((ranks * sorted_actual).sum() - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count)
            auc = auc.item()
            auc_values.append(auc)
        else:
            auc = float('nan')

        classwise_metrics[class_index] = {
            'AUC': auc,
            'Accuracy': (true_positive + true_negative) / len(labels),
            'Balanced Accuracy': (recall + specificity) / 2,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Specificity': specificity,
            'Support': positive_count,
        }

    recalls = [class_metrics['Recall'] for class_metrics in classwise_metrics.values()]
    metrics = {
        'overall': {
            'AUC': sum(auc_values) / len(auc_values) if auc_values else float('nan'),
            'Accuracy': (predictions == labels).float().mean().item(),
            'Balanced Accuracy': sum(recalls) / len(recalls),
            'Precision': sum(class_metrics['Precision'] for class_metrics in classwise_metrics.values()) / num_classes,
            'Recall': sum(recalls) / num_classes,
            'F1': sum(class_metrics['F1'] for class_metrics in classwise_metrics.values()) / num_classes,
        },
        'classwise': classwise_metrics,
    }

    print('Overall metrics:')
    for name, value in metrics['overall'].items():
        print(f'  {name}: {value:.4f}')
    print('Classwise metrics:')
    for class_index, class_metrics in classwise_metrics.items():
        formatted_metrics = ', '.join(f'{name}={value:.4f}' for name, value in class_metrics.items() if name != 'Support')
        print(f'  Class {class_index} (support={class_metrics["Support"]}): {formatted_metrics}')

    return metrics

def main():
    torch.manual_seed(42)
    DATASET = helpers.Dataset(*helpers.BLOOD)
    SIZE = None
    train_dataloader, valid_dataloader, test_dataloader = dataloader.get_dataloaders(DATASET.get_load_data_func(size=SIZE),
                                                                                     batch_size=32, 
                                                                                     shuffle=True, 
                                                                                     num_workers=4)

    model = BaselineModel.BaselineModel(num_classes=DATASET.num_classes, num_channels=3)

    tag = timestamp.get_timestamp() + f"_{DATASET.name}_{model.name}"

    train_losses, valid_losses = train_model(model, train_dataloader, valid_dataloader, 
                                             num_epochs=100, learning_rate=0.001, device='cuda', 
                                             tag=tag)

    results_dir = r'C:\Users\tomst\multitask-ml-project\src\project\experiment1\results'
    os.makedirs(results_dir, exist_ok=True)
    plotting.plot_loss(train_losses, valid_losses, title='Loss over Epochs', root=results_dir, model_name=model.name, tag=tag)

    metrics = evaluate_model(model, test_dataloader, device='cuda')

    json_metrics = json.dumps(metrics, indent=4)
    with open(f'{results_dir}\\evaluation_metrics_{tag}.json', 'w') as f:
        f.write(json_metrics)

if __name__ == "__main__":
    main()