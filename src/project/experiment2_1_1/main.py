from functools import partial
import argparse
from pathlib import Path
import re
from typing import Any, Dict, Union
import json, os
import torch
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv
from project.myModules import timestamp, plotting, stattest
from project.myModules.helpers import *
from project.experiment2_1_1.BaselineModel import BaselineModel
from project.experiment2_1_1.GrowingModel import GrowingModel
from project.experiment2_1_1.Classifier import ClassifierHeads
from project.experiment2_1_1.PrettyResults import make_pretty_results
from project.experiment2_1_1 import multidataloader

TASKS = {
    "Blood": (8, 3, multidataloader.load_blood_data),
    "Breast": (2, 1, multidataloader.load_breast_data),
    "Derma": (7, 3, multidataloader.load_derma_data),
    "Path": (9, 3, multidataloader.load_path_data),
    "Tissue": (8, 1, multidataloader.load_tissue_data),
}
TASK_GROUPS = (("Blood", "Derma", "Path"), ("Breast", "Tissue"))

def train_model(model: torch.nn.Module,
                classifiers: ClassifierHeads,
                train_dataloader: torch.utils.data.DataLoader,
                valid_dataloaders: dict[str, torch.utils.data.DataLoader],
                num_epochs: int = 100,
                learning_rate: float = 0.001,
                device: str = 'cuda',
                tag: str = "run",
                log_path: str = "./runs") -> Union[list, list]:
    """Train one shared backbone against each task's independent classifier."""
    classifiers.to(torch.device(device))
    print(f"Training on device: {device}")
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(classifiers.parameters()),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=14,
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_valid_loss = float('inf')
    best_epoch = 0
    plateau = 5
    train_losses = []
    valid_losses = []
    tensorboard_writer = SummaryWriter(log_dir=f'{log_path}/{tag}')

    for epoch in range(num_epochs):
        train_batch_count = len(train_dataloader)
        valid_batch_count = sum(len(loader) for loader in valid_dataloaders.values())
        model.train()
        classifiers.train()
        train_loss_total = 0.0
        train_sample_count = 0
        for batch_number, (inputs, labels, tasks) in enumerate(train_dataloader, start=1):
            inputs = inputs.to(device)
            labels = labels.to(device).view(-1).long()
            features = model(inputs)
            task_losses = []
            for task in dict.fromkeys(tasks):
                sample_indices = torch.tensor(
                    [index for index, sample_task in enumerate(tasks) if sample_task == task],
                    device=labels.device,
                )
                task_features = features.index_select(0, sample_indices)
                task_labels = labels.index_select(0, sample_indices)
                task_outputs = classifiers(task, task_features)
                task_losses.append(criterion(task_outputs, task_labels))

            loss = torch.stack(task_losses).mean()
            optimizer.zero_grad()
            loss.backward()

            with torch.no_grad():
                for module in model.modules():
                    if hasattr(module, 'grad_mask') and module.weight.grad is not None:
                        module.weight.grad *= module.grad_mask

            optimizer.step()
            train_loss_total += loss.item() * inputs.size(0)
            train_sample_count += inputs.size(0)
            task_names = ",".join(dict.fromkeys(tasks))
            print(
                f"Epoch {epoch + 1}/{num_epochs} | Train batch {batch_number}/{train_batch_count} "
                f"| Tasks {task_names} | Loss {loss.item():.4f}".ljust(100),
                end="\r",
                flush=True,
            )

        train_loss = train_loss_total / train_sample_count
        model.eval()
        classifiers.eval()
        valid_loss_total = 0.0
        valid_sample_count = 0
        valid_loss_by_task = {}
        valid_batch_number = 0
        with torch.no_grad():
            for task, valid_dataloader in valid_dataloaders.items():
                task_loss_total = 0.0
                task_sample_count = 0
                for inputs, labels in valid_dataloader:
                    valid_batch_number += 1
                    inputs = inputs.to(device)
                    labels = labels.to(device).view(-1).long()
                    outputs = classifiers(task, model(inputs))
                    loss = criterion(outputs, labels)
                    valid_loss_total += loss.item() * inputs.size(0)
                    valid_sample_count += inputs.size(0)
                    task_loss_total += loss.item() * inputs.size(0)
                    task_sample_count += inputs.size(0)
                    print(
                        f"Epoch {epoch + 1}/{num_epochs} | Validation batch "
                        f"{valid_batch_number}/{valid_batch_count} ({task})".ljust(100),
                        end="\r",
                        flush=True,
                    )
                valid_loss_by_task[task] = task_loss_total / task_sample_count

        valid_loss = valid_loss_total / valid_sample_count
        tensorboard_writer.add_scalar('Loss/Train', train_loss, epoch)
        tensorboard_writer.add_scalars(
            'Loss/Validation',
            {'Overall': valid_loss, **valid_loss_by_task},
            epoch,
        )
        previous_learning_rate = optimizer.param_groups[0]['lr']
        scheduler.step(valid_loss)
        current_learning_rate = optimizer.param_groups[0]['lr']
        tensorboard_writer.add_scalar('Learning Rate', current_learning_rate, epoch)
        if current_learning_rate < previous_learning_rate:
            print(f"Learning rate reduced to {current_learning_rate:.6g} after validation plateau.")
        print(f'Epoch {epoch+1}/{num_epochs} complete | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}'.ljust(100))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch

        if epoch - best_epoch >= plateau and epoch < int(num_epochs * 8):
            if model.grow() != -1:
                print(f"\nModel grown at epoch {epoch+1}. Best validation loss: {best_valid_loss:.4f} at epoch {best_epoch+1}.")
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
                   classifiers: ClassifierHeads,
                   task: str,
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
    classifiers.to(torch.device(device))
    classifiers.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []
    sample_predictions = []
    next_sample_index = 0

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs: torch.Tensor = inputs.to(device)
            labels: torch.Tensor = labels.to(device).view(-1).long()
            outputs: torch.Tensor = classifiers(task, model(inputs))
            predictions = outputs.argmax(dim=1)

            all_labels.append(labels.cpu())
            all_predictions.append(predictions.cpu())
            all_probabilities.append(outputs.cpu())
            batch_labels = labels.cpu().tolist()
            batch_predictions = predictions.cpu().tolist()
            batch_scores = outputs.cpu().tolist()
            sample_predictions.extend(
                {
                    'sample_index': next_sample_index + offset,
                    'true_label': true_label,
                    'predicted_label': predicted_label,
                    'class_scores': scores,
                }
                for offset, (true_label, predicted_label, scores) in enumerate(
                    zip(batch_labels, batch_predictions, batch_scores)
                )
            )
            next_sample_index += len(batch_labels)

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
        'sample_score_type': 'logits',
        'sample_predictions': sample_predictions,
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

    print(f'{task} overall metrics:')
    for name, value in metrics['overall'].items():
        print(f'  {name}: {value:.4f}')
    print(f'{task} classwise metrics:')
    for class_index, class_metrics in classwise_metrics.items():
        formatted_metrics = ', '.join(f'{name}={value:.4f}' for name, value in class_metrics.items() if name != 'Support')
        print(f'  Class {class_index} (support={class_metrics["Support"]}): {formatted_metrics}')

    return metrics

def run(model_class: torch.nn.Module, task_names: tuple[str, ...], size: int):
    torch.manual_seed(42)
    device='cuda:1'
    num_workers = 0 if os.name == 'nt' and size == 224 else 4
    load_dotenv()
    proj_path = os.getenv('PROJ_PATH')
    data_path = os.path.join(proj_path, 'src', 'project', 'data')
    log_path = os.path.join(proj_path, 'src', 'project', 'experiment2_1_1', 'runs')

    data_load_fns = {
        task: partial(TASKS[task][2], path=data_path, size=size)
        for task in task_names
    }
    train_dataloader, valid_dataloaders, test_dataloaders = multidataloader.get_multitask_dataloaders(
        data_load_fns,
        batch_size=30 if size==224 else 60,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True
    )

    num_channels = TASKS[task_names[0]][1]
    model: torch.nn.Module = model_class(num_channels=num_channels, size=size, device=device)
    classifiers = ClassifierHeads({task: TASKS[task][0] for task in task_names})
    run_timestamp = timestamp.get_timestamp()
    tag = f"{run_timestamp}_{'-'.join(task_names)}_{size}_{model.name}"

    train_losses, valid_losses = train_model(
        model,
        classifiers,
        train_dataloader,
        valid_dataloaders,
        num_epochs=100,
        learning_rate=0.001,
        device=device,
        tag=tag,
        log_path=log_path,
    )

    results_dir = os.path.join(proj_path, 'src', 'project', 'experiment2_1_1', 'results')
    os.makedirs(results_dir, exist_ok=True)
    plotting.plot_loss(train_losses, valid_losses, title='Loss over Epochs', root=results_dir, model_name=model.name, tag=tag)

    task_metrics = {}
    for task, test_dataloader in test_dataloaders.items():
        metrics = evaluate_model(model, classifiers, task, test_dataloader, device=device)
        task_metrics[task] = metrics
        result_tag = f"{run_timestamp}_{task}_{size}_{model.name}"
        json_metrics = json.dumps(metrics, indent=4)
        with open(f'{results_dir}\\evaluation_metrics_{result_tag}.json', 'w') as result_file:
            result_file.write(json_metrics)

    metric_names = next(iter(task_metrics.values()))['overall']
    print(f'Combined test performance (equal-task average across {len(task_metrics)} tasks):')
    for metric in metric_names:
        average = sum(metrics['overall'][metric] for metrics in task_metrics.values()) / len(task_metrics)
        print(f'  {metric}: {average:.4f}')

    model_dir = os.path.join(proj_path, 'src', 'project', 'experiment2_1_1', 'models')
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), f'{model_dir}\\backbone_weights_{tag}.pth')
    for task, classifier in classifiers.heads.items():
        head_tag = f"{run_timestamp}_{task}_{size}_{model.name}"
        torch.save(classifier.state_dict(), f'{model_dir}\\classifier_weights_{head_tag}.pth')


def test_pretrained_model(
      model_class: type[torch.nn.Module],
    task_names: tuple[str, ...],
    size: int,
    backbone_weights_path: str | Path,
    device: str | None = None,
    data_path: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> dict[str, Dict[str, Any]]:
    """Evaluate a saved model configuration without running training."""
    weights_path = Path(backbone_weights_path).expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Backbone weights not found: {weights_path}")

    match = re.fullmatch(
        r"backbone_weights_(?P<timestamp>\d{8}-\d{6})_"
        r"(?P<tasks>[A-Za-z]+(?:-[A-Za-z]+)*)_"
        r"(?P<checkpoint_size>\d+)_(?P<checkpoint_model>BaselineModel|GrowingModel)",
        weights_path.stem,
    )
    if match is None:
        raise ValueError(
            "Expected checkpoint name like "
            "backbone_weights_YYYYMMDD-HHMMSS_Blood-Derma-Path_224_GrowingModel.pth"
        )

    checkpoint_timestamp = match.group("timestamp")
    checkpoint_tasks = tuple(match.group("tasks").split("-"))
    if checkpoint_tasks != task_names:
        raise ValueError(f"Checkpoint tasks {checkpoint_tasks} do not match requested tasks {task_names}")
    if int(match.group("checkpoint_size")) != size:
        raise ValueError(f"Checkpoint image size does not match requested size {size}")
    model_name = model_class.__name__
    if match.group("checkpoint_model") != model_name:
        raise ValueError(f"Checkpoint model {match.group('checkpoint_model')} does not match {model_name}")
    unknown_tasks = set(task_names) - TASKS.keys()
    if unknown_tasks:
        raise ValueError(f"Unknown task(s) in checkpoint name: {', '.join(sorted(unknown_tasks))}")
    if len({TASKS[task][1] for task in task_names}) != 1:
        raise ValueError("Tasks in one checkpoint must use the same number of input channels")

    load_dotenv()
    project_root = Path(os.getenv('PROJ_PATH') or Path(__file__).resolve().parents[3])
    resolved_data_path = Path(data_path) if data_path is not None else project_root / 'src' / 'project' / 'data'
    resolved_results_dir = Path(results_dir) if results_dir is not None else project_root / 'src' / 'project' / 'experiment2_1_1' / 'results'
    resolved_device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 0
    data_load_fns = {
        task: partial(TASKS[task][2], path=str(resolved_data_path), size=size)
        for task in task_names
    }
    test_dataloaders = multidataloader.get_multitask_test_loaders(
        data_load_fns,
        batch_size=30 if size == 224 else 60,
        num_workers=num_workers,
        persistent_workers=True,
    )

    model = model_class(
        num_channels=TASKS[task_names[0]][1],
        size=size,
        device=resolved_device,
    )
    model.load_state_dict(torch.load(weights_path, map_location=resolved_device, weights_only=True))

    classifiers = ClassifierHeads({task: TASKS[task][0] for task in task_names}).to(resolved_device)
    for task in task_names:
        head_name = f"classifier_weights_{checkpoint_timestamp}_{task}_{size}_{model_name}.pth"
        head_path = weights_path.parent / head_name
        if not head_path.is_file():
            raise FileNotFoundError(f"Classifier weights not found: {head_path}")
        classifiers.heads[task].load_state_dict(
            torch.load(head_path, map_location=resolved_device, weights_only=True)
        )

    evaluation_timestamp = timestamp.get_timestamp()
    task_metrics = {}
    for task, test_dataloader in test_dataloaders.items():
        metrics = evaluate_model(model, classifiers, task, test_dataloader, device=resolved_device)
        metrics['checkpoint'] = str(weights_path)
        task_metrics[task] = metrics
        result_tag = f"{evaluation_timestamp}_{task}_{size}_{model_name}"
        resolved_results_dir.mkdir(parents=True, exist_ok=True)
        result_path = resolved_results_dir / f"evaluation_metrics_{result_tag}.json"
        result_path.write_text(json.dumps(metrics, indent=4), encoding='utf-8')
        print(f"Wrote test results to {result_path}")

    metric_names = next(iter(task_metrics.values()))['overall']
    print(f'Combined test performance (equal-task average across {len(task_metrics)} tasks):')
    for metric in metric_names:
        average = sum(metrics['overall'][metric] for metrics in task_metrics.values()) / len(task_metrics)
        print(f'  {metric}: {average:.4f}')
    return task_metrics

def main(train: bool = True):
    model_dir = Path(__file__).resolve().parent / 'models'
    results_dir = Path(__file__).resolve().parent / 'results'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tested_count = 0
    for size in (28, 224):
        for task_group in TASK_GROUPS:
            for model_class in (BaselineModel, GrowingModel):
                if train:
                    run(model_class, task_group, size)
                else:
                    pattern = (
                        f"backbone_weights_*_{'-'.join(task_group)}_"
                        f"{size}_{model_class.__name__}.pth"
                    )
                    checkpoint_paths = sorted(model_dir.glob(pattern))
                    for checkpoint_path in checkpoint_paths:
                        print(f"Testing checkpoint: {checkpoint_path.name}")
                        test_pretrained_model(
                            model_class,
                            task_group,
                            size,
                            checkpoint_path,
                            device=device,
                            results_dir=results_dir,
                        )
                        tested_count += 1

    if not train and tested_count == 0:
        raise FileNotFoundError(f"No saved backbone checkpoints found in {model_dir}")
    print(f"Tested {tested_count} saved model checkpoint(s).")

if __name__ == "__main__":
    main(train=True)
    make_pretty_results()
    stattest.main(default_results_dir=os.path.abspath("results"))
