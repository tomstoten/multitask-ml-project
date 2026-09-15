from typing import Any, Dict, Union
import json, os
import torch
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv
from project.myModules import dataloader, timestamp, plotting
from project.myModules.helpers import *
from project.experiment1.BaselineModel import BaselineModel
from project.experiment1.GrowingModel import GrowingModel
from project.experiment1.Classifier import Classifier
from project.experiment1.PrettyResults import make_pretty_results

def train_model(model: torch.nn.Module, 
                classifier: torch.nn.Module,
                train_dataloader: torch.utils.data.DataLoader, 
                valid_dataloader: torch.utils.data.DataLoader, 
                num_epochs: int = 100, 
                learning_rate: float = 0.001, 
                device: str = 'cuda',
                tag: str = "run",
                log_path: str = "./runs") -> Union[list, list]:
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
        log_path: Path to the logging directory
    """
    classifier.to(torch.device(device))
    print(f"Training on device: {device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    best_valid_loss = float('inf')
    best_epoch = 0
    plateau = 5

    train_losses = []
    valid_losses = []

    tensorboard_writer = SummaryWriter(log_dir=f'{log_path}/{tag}')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_dataloader:
            inputs: torch.Tensor = inputs.to(device)
            labels: torch.Tensor = labels.to(device)
            # make labels one-hot encoded for CrossEntropyLoss
            labels = torch.nn.functional.one_hot(labels, num_classes=classifier.out_features).float().squeeze()

            optimizer.zero_grad()
            outputs: torch.Tensor = classifier(model(inputs))
            
            loss: torch.Tensor = criterion(outputs, labels)
            loss.backward()

            with torch.no_grad():
                for module in model.modules():
                    if hasattr(module, 'grad_mask'):
                        module.weight.grad *= module.grad_mask
            
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
                labels = torch.nn.functional.one_hot(labels, num_classes=classifier.out_features).float().squeeze()
                
                outputs: torch.Tensor = classifier(model(inputs))
                loss: torch.Tensor = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)

        valid_loss /= len(valid_dataloader.dataset)

        tensorboard_writer.add_scalar('Loss/Train', train_loss, epoch)
        tensorboard_writer.add_scalar('Loss/Validation', valid_loss, epoch)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}', end='\r')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch

        if epoch - best_epoch >= plateau and epoch < int(num_epochs * 8):
            if model.grow() != -1:
                print(f"\nModel grown at epoch {epoch+1}. Best validation loss: {best_valid_loss:.4f} at epoch {best_epoch+1}.")
            # else:
            #     print(f"\nEarly stopping at epoch {epoch+1}. Best validation loss: {best_valid_loss:.4f} at epoch {best_epoch+1}.")
            #     break
            best_epoch = epoch
            best_valid_loss = valid_loss

        if epoch == int(num_epochs * 8):
            while model.grow() != -1:
                pass
            else:
                print("Model fully grown already.")

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

    tensorboard_writer.close()
    print("\nTraining complete.")
    return train_losses, valid_losses

def evaluate_model(model: torch.nn.Module,
                   classifier: torch.nn.Module,
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
    model.eval()
    classifier.to(torch.device(device))
    classifier.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs: torch.Tensor = inputs.to(device)
            labels: torch.Tensor = labels.to(device).view(-1).long()
            outputs: torch.Tensor = classifier(model(inputs))
            predictions = outputs.argmax(dim=1)

            all_labels.append(labels.cpu())
            all_predictions.append(predictions.cpu())
            all_probabilities.append(outputs.cpu())

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

def run(model_class: torch.nn.Module, dataset: Dataset, size: int | None):
    torch.manual_seed(42)
    device='cuda'
    num_workers = 0 if os.name == 'nt' and size == 224 and (dataset.name == 'Path' or dataset.name == "Tissue") else 6
    load_dotenv()
    proj_path = os.getenv('PROJ_PATH')
    data_path = os.path.join(proj_path, 'src', 'project', 'data')
    log_path = os.path.join(proj_path, 'src', 'project', 'experiment1', 'runs')
    
    train_dataloader, valid_dataloader, test_dataloader = dataloader.get_dataloaders(dataset.get_load_data_func(
                                                                                        path=data_path,
                                                                                        size=size),
                                                                                     batch_size=32, 
                                                                                     shuffle=True, 
                                                                                     num_workers=num_workers)

    model: torch.nn.Module = model_class(num_channels=dataset.num_channels, size=size, device=device)
    classifier = Classifier(num_classes=dataset.num_classes)

    tag = timestamp.get_timestamp() + f"_{dataset.name}{f"_{size}" if size is not None else ''}_{model.name}"

    train_losses, valid_losses = train_model(model, classifier, train_dataloader, valid_dataloader, 
                                             num_epochs=100, learning_rate=0.001, device=device, 
                                             tag=tag, log_path=log_path)

    results_dir = os.path.join(proj_path, 'src', 'project', 'experiment1', 'results')
    os.makedirs(results_dir, exist_ok=True)
    plotting.plot_loss(train_losses, valid_losses, title='Loss over Epochs', root=results_dir, model_name=model.name, tag=tag)

    metrics = evaluate_model(model, classifier, test_dataloader, device=device)

    json_metrics = json.dumps(metrics, indent=4)
    with open(f'{results_dir}\\evaluation_metrics_{tag}.json', 'w') as f:
        f.write(json_metrics)

    # save model and classifier state dicts/weights
    model_dir = os.path.join(proj_path, 'src', 'project', 'experiment1', 'models')
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), f'{model_dir}\\model_weights_{tag}.pth')
    torch.save(classifier.state_dict(), f'{model_dir}\\classifier_weights_{tag}.pth')

def main():
    # for model_class in [BaselineModel, GrowingModel]:
    #     run(model_class, Dataset(*Dataset.TISSUE), size=None)
    
    size=224
    for model_class in [BaselineModel]:
        for dataset in (Dataset(*Dataset.PATH), Dataset(*Dataset.TISSUE)):
            run(model_class, dataset, size=size)
            
    size=224
    for model_class in [GrowingModel]:
        for dataset in (Dataset(*Dataset.BLOOD), Dataset(*Dataset.BREAST), Dataset(*Dataset.DERMA), Dataset(*Dataset.PATH), Dataset(*Dataset.TISSUE)):
            run(model_class, dataset, size=size)

if __name__ == "__main__":
    main()
    make_pretty_results()