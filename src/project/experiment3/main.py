import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from functools import partial
from pathlib import Path
import re
import copy
from typing import Any, Dict, Union
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv
from project.myModules import timestamp, plotting, stattest
from project.myModules.helpers import *
from project.experiment3.BaselineModel import BaselineModel
from project.experiment3.GrowingModel import GrowingModel
from project.experiment3.Classifier import ClassifierHeads
from project.experiment3.PrettyResults import make_pretty_results
from project.experiment3 import multidataloader

TASKS = {
    "Blood": (8, 3, multidataloader.load_blood_data),
    "Breast": (2, 1, multidataloader.load_breast_data),
    "Derma": (7, 3, multidataloader.load_derma_data),
    "Path": (9, 3, multidataloader.load_path_data),
    "Tissue": (8, 1, multidataloader.load_tissue_data),
    "Adrenal": (2, 1, multidataloader.load_adrenal_data),
    "Chest": (14, 1, multidataloader.load_chest_data),
    "Fracture": (3, 1, multidataloader.load_fracture_data),
    "Nodule": (2, 1, multidataloader.load_nodule_data),
    "OCT": (4, 1, multidataloader.load_oct_data),
    "Organ3D": (11, 1, multidataloader.load_organ3d_data),
    "OrganA": (11, 1, multidataloader.load_organa_data),
    "Pneumonia": (2, 1, multidataloader.load_pneumonia_data),
    "Retina": (5, 3, multidataloader.load_retina_data),
    "Synapse": (2, 1, multidataloader.load_synapse_data),
    "Vessel": (2, 1, multidataloader.load_vessel_data),
}
TASK_GROUPS = (
    ("Blood", "Derma", "Path", "Retina"),
    ("Breast", "OCT", "OrganA", "Pneumonia", "Tissue"),
    ("Adrenal", "Fracture", "Nodule", "Organ3D", "Synapse", "Vessel"),
    ("Chest",),
)
TASK_GROUP_SIZES = ((28, 224), (28, 224), (28, 64), (28, 224))
VOLUME_TASKS = {"Adrenal", "Fracture", "Nodule", "Organ3D", "Synapse", "Vessel"}
MULTI_LABEL_TASKS = {"Chest"}
TEMPORARY_TRAIN_START_INDEX = 2


def _batch_size_for_tasks(task_names: tuple[str, ...], size: int) -> int:
    if any(task in VOLUME_TASKS for task in task_names):
        return 2 if size == 64 else 4
    return 30 if size == 224 else 60


def _forward_for_training(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    if inputs.shape[0] != 1:
        return model(inputs)

    batch_norms = [
        module for module in model.modules()
        if isinstance(module, torch.nn.BatchNorm1d)
    ]
    training_states = [module.training for module in batch_norms]
    try:
        for module in batch_norms:
            module.training = False
        return model(inputs)
    finally:
        for module, was_training in zip(batch_norms, training_states):
            module.training = was_training


def _forward_with_task_batch_statistics(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Use task-local BatchNorm statistics for task-homogeneous evaluation batches."""
    batch_norms = [
        module for module in model.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]
    saved_state = [
        (
            module.training,
            module.momentum,
            module.running_mean.clone() if module.running_mean is not None else None,
            module.running_var.clone() if module.running_var is not None else None,
            module.num_batches_tracked.clone() if module.num_batches_tracked is not None else None,
        )
        for module in batch_norms
    ]
    for module in batch_norms:
        if isinstance(module, torch.nn.BatchNorm1d) and inputs.shape[0] < 2:
            continue
        module.training = True
        module.momentum = 0.0

    try:
        return model(inputs)
    finally:
        for module, (was_training, momentum, running_mean, running_var, batches_tracked) in zip(batch_norms, saved_state):
            module.training = was_training
            module.momentum = momentum
            if running_mean is not None:
                module.running_mean.copy_(running_mean)
            if running_var is not None:
                module.running_var.copy_(running_var)
            if batches_tracked is not None:
                module.num_batches_tracked.copy_(batches_tracked)

def train_model(model: torch.nn.Module,
                classifiers: ClassifierHeads,
                train_dataloaders: dict[str, torch.utils.data.DataLoader],
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
    scheduler_patience = 14
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=scheduler_patience,
    )
    criterion = torch.nn.CrossEntropyLoss()
    multi_label_criterion = torch.nn.BCEWithLogitsLoss()

    best_valid_loss = float('inf')
    previous_phases_best_loss = float('inf')
    growth_phase_best_loss = float('inf')
    growth_phase_best_epoch = -1
    early_stopping_epoch = -1
    growth_patience = 20
    early_stopping_patience = max(
        40 if hasattr(model, "grow") else 20,
        scheduler_patience + 10,
    )
    growth_cooldown = 20
    last_growth_epoch = -growth_cooldown
    best_model_state = copy.deepcopy(model.state_dict())
    best_classifiers_state = copy.deepcopy(classifiers.state_dict())
    best_growth_metadata = (
        copy.deepcopy(model.get_growth_metadata())
        if hasattr(model, "get_growth_metadata") else None
    )
    train_losses = []
    valid_losses = []
    tensorboard_writer = SummaryWriter(log_dir=f'{log_path}/{tag}')

    if not train_dataloaders or any(len(loader) == 0 for loader in train_dataloaders.values()):
        raise ValueError("Every training task must have at least one batch")
    batches_per_task = min(len(loader) for loader in train_dataloaders.values())

    epoch = 0
    while epoch < num_epochs:
        task_schedule = [
            task
            for task in train_dataloaders
            for _ in range(batches_per_task)
        ]
        schedule_order = torch.randperm(len(task_schedule)).tolist()
        train_iterators = {task: iter(loader) for task, loader in train_dataloaders.items()}
        train_batch_count = len(task_schedule)
        valid_batch_count = sum(len(loader) for loader in valid_dataloaders.values())
        model.train()
        classifiers.train()
        train_loss_by_task = {task: [0.0, 0] for task in train_dataloaders}
        for batch_number, schedule_index in enumerate(schedule_order, start=1):
            task = task_schedule[schedule_index]
            inputs, labels = next(train_iterators[task])
            inputs = inputs.to(device)
            labels = labels.to(device)
            if task in MULTI_LABEL_TASKS:
                labels = labels.float()
            else:
                labels = labels.view(-1).long()
            outputs = classifiers(task, _forward_for_training(model, inputs))
            loss = (
                multi_label_criterion(outputs, labels)
                if task in MULTI_LABEL_TASKS else criterion(outputs, labels)
            )
            optimizer.zero_grad()
            loss.backward()

            with torch.no_grad():
                for module in model.modules():
                    if hasattr(module, 'grad_mask') and module.weight.grad is not None:
                        module.weight.grad *= module.grad_mask

            optimizer.step()
            train_loss_by_task[task][0] += loss.item() * inputs.size(0)
            train_loss_by_task[task][1] += inputs.size(0)
            print(
                f"Epoch {epoch + 1}/{num_epochs} | Train batch {batch_number}/{train_batch_count} "
                f"| Task {task} | Loss {loss.item():.4f}".ljust(100),
                end="\r",
                flush=True,
            )

        train_loss_by_task = {
            task: total / samples
            for task, (total, samples) in train_loss_by_task.items()
            if samples
        }
        train_loss = sum(train_loss_by_task.values()) / len(train_loss_by_task)
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
                    labels = labels.to(device)
                    if task in MULTI_LABEL_TASKS:
                        labels = labels.float()
                    else:
                        labels = labels.view(-1).long()
                    outputs = classifiers(task, _forward_with_task_batch_statistics(model, inputs))
                    loss = (
                        multi_label_criterion(outputs, labels)
                        if task in MULTI_LABEL_TASKS else criterion(outputs, labels)
                    )
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
        macro_valid_loss = sum(valid_loss_by_task.values()) / len(valid_loss_by_task)
        tensorboard_writer.add_scalars('Loss/Train', {'Overall': train_loss, **train_loss_by_task}, epoch)
        tensorboard_writer.add_scalars(
            'Loss/Validation',
            {'Overall': valid_loss, 'Macro Task Average': macro_valid_loss, **valid_loss_by_task},
            epoch,
        )
        growth_complete = getattr(model, "growth_complete", True)
        previous_learning_rate = optimizer.param_groups[0]['lr']
        if growth_complete:
            scheduler.step(macro_valid_loss)
        current_learning_rate = optimizer.param_groups[0]['lr']
        tensorboard_writer.add_scalar('Learning Rate', current_learning_rate, epoch)
        if current_learning_rate < previous_learning_rate:
            print(f"Learning rate reduced to {current_learning_rate:.6g} after validation plateau.")
        print(
            f'Epoch {epoch+1}/{num_epochs} complete | Train Loss: {train_loss:.4f} '
            f'| Macro Valid Loss: {macro_valid_loss:.4f} | Sample-weighted Valid Loss: {valid_loss:.4f}'.ljust(130)
        )

        if macro_valid_loss < best_valid_loss:
            best_valid_loss = macro_valid_loss
            early_stopping_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_classifiers_state = copy.deepcopy(classifiers.state_dict())
            if hasattr(model, "get_growth_metadata"):
                best_growth_metadata = copy.deepcopy(model.get_growth_metadata())

        if macro_valid_loss < growth_phase_best_loss:
            growth_phase_best_loss = macro_valid_loss
            growth_phase_best_epoch = epoch

        cooldown_complete = epoch - last_growth_epoch >= growth_cooldown
        surpassed_previous_phases = growth_phase_best_loss < previous_phases_best_loss
        plateau_reached = epoch - growth_phase_best_epoch >= growth_patience
        if not growth_complete and cooldown_complete and surpassed_previous_phases and plateau_reached:
            if model.grow() == -1:
                print("Model reported no remaining growth phases.")
            else:
                last_growth_epoch = epoch
                early_stopping_epoch = epoch
                previous_phases_best_loss = min(previous_phases_best_loss, growth_phase_best_loss)
                growth_phase_best_loss = float('inf')
                growth_phase_best_epoch = epoch
                print(
                    f"\nModel grown to {getattr(model, 'growth_proportion', 'next phase')} at epoch "
                    f"{epoch + 1}; the phase beat the previous best and then had "
                    f"{growth_patience} epochs without improvement."
                )

        train_losses.append(train_loss)
        valid_losses.append(macro_valid_loss)

        if (
            getattr(model, "growth_complete", True)
            and epoch - early_stopping_epoch >= early_stopping_patience
        ):
            print(
                f"\nEarly stopping at epoch {epoch + 1}; no validation improvement or model growth since "
                f"epoch {early_stopping_epoch + 1}."
            )
            break

        epoch += 1

    tensorboard_writer.close()
    model.load_state_dict(best_model_state)
    if best_growth_metadata is not None:
        model.restore_growth_metadata(best_growth_metadata)
    classifiers.load_state_dict(best_classifiers_state)
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
            labels: torch.Tensor = labels.to(device)
            if task in MULTI_LABEL_TASKS:
                labels = labels.float()
            else:
                labels = labels.view(-1).long()
            outputs: torch.Tensor = classifiers(task, _forward_with_task_batch_statistics(model, inputs))
            if task in MULTI_LABEL_TASKS:
                probabilities = outputs.sigmoid()
                predictions = probabilities >= 0.5
            else:
                probabilities = outputs
                predictions = outputs.argmax(dim=1)

            all_labels.append(labels.cpu())
            all_predictions.append(predictions.cpu())
            all_probabilities.append(probabilities.cpu())
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
        if task in MULTI_LABEL_TASKS:
            actual = labels[:, class_index] >= 0.5
            predicted = predictions[:, class_index]
        else:
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
            'Accuracy': (true_positive + true_negative) / actual.numel(),
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
            'Accuracy': (
                (predictions == labels.bool()).float().mean().item()
                if task in MULTI_LABEL_TASKS
                else (predictions == labels).float().mean().item()
            ),
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

def run(model_class: torch.nn.Module, task_names: tuple[str, ...], size: int, device: torch.device = 'cuda'):
    torch.manual_seed(42)
    is_3d_run = any(task in VOLUME_TASKS for task in task_names)
    num_workers = 0 if is_3d_run or (os.name == 'nt' and size == 224) else 4
    load_dotenv()
    proj_path = os.getenv('PROJ_PATH')
    data_path = os.path.join(proj_path, 'src', 'project', 'data')
    log_path = os.path.join(proj_path, 'src', 'project', 'experiment3', 'runs')

    data_load_fns = {
        task: partial(TASKS[task][2], path=data_path, size=size)
        for task in task_names
    }
    train_dataloaders, valid_dataloaders, test_dataloaders = multidataloader.get_multitask_dataloaders(
        data_load_fns,
        batch_size=_batch_size_for_tasks(task_names, size),
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True
    )

    num_channels = TASKS[task_names[0]][1]
    task_spatial_dims = {3 if task in VOLUME_TASKS else 2 for task in task_names}
    if len(task_spatial_dims) != 1:
        raise ValueError("Tasks in one run must have the same spatial dimensionality")
    spatial_dims = task_spatial_dims.pop()
    model: torch.nn.Module = model_class(
        num_channels=num_channels,
        size=size,
        device=device,
        spatial_dims=spatial_dims,
    )
    classifiers = ClassifierHeads({task: TASKS[task][0] for task in task_names})
    run_timestamp = timestamp.get_timestamp()
    tag = f"{run_timestamp}_{'-'.join(task_names)}_{size}_{model.name}"

    train_losses, valid_losses = train_model(
        model,
        classifiers,
        train_dataloaders,
        valid_dataloaders,
        num_epochs=2000,
        learning_rate=0.001,
        device=device,
        tag=tag,
        log_path=log_path,
    )

    results_dir = os.path.join(proj_path, 'src', 'project', 'experiment3', 'results')
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

    model_dir = os.path.join(proj_path, 'src', 'project', 'experiment3', 'models')
    os.makedirs(model_dir, exist_ok=True)
    backbone_path = f'{model_dir}\\backbone_weights_{tag}.pth'
    if hasattr(model, "get_growth_metadata"):
        torch.save(
            {
                "state_dict": model.state_dict(),
                "growth_metadata": model.get_growth_metadata(),
            },
            backbone_path,
        )
    else:
        torch.save(model.state_dict(), backbone_path)
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
    if len({3 if task in VOLUME_TASKS else 2 for task in task_names}) != 1:
        raise ValueError("Tasks in one checkpoint must use the same spatial dimensionality")

    load_dotenv()
    project_root = Path(os.getenv('PROJ_PATH') or Path(__file__).resolve().parents[3])
    resolved_data_path = Path(data_path) if data_path is not None else project_root / 'src' / 'project' / 'data'
    resolved_results_dir = Path(results_dir) if results_dir is not None else project_root / 'src' / 'project' / 'experiment3' / 'results'
    resolved_device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 0
    data_load_fns = {
        task: partial(TASKS[task][2], path=str(resolved_data_path), size=size)
        for task in task_names
    }
    test_dataloaders = multidataloader.get_multitask_test_loaders(
        data_load_fns,
        batch_size=_batch_size_for_tasks(task_names, size),
        num_workers=num_workers,
        persistent_workers=True,
    )

    model = model_class(
        num_channels=TASKS[task_names[0]][1],
        size=size,
        device=resolved_device,
        spatial_dims=3 if task_names[0] in VOLUME_TASKS else 2,
    )
    checkpoint = torch.load(weights_path, map_location=resolved_device, weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
        if "growth_metadata" in checkpoint and hasattr(model, "restore_growth_metadata"):
            model.restore_growth_metadata(checkpoint["growth_metadata"])
    else:
        model.load_state_dict(checkpoint)

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
    device = 'cuda' if torch.cuda.device_count() > 1 else 'cuda' if torch.cuda.is_available() else 'cpu'
    tested_count = 0
    group_start = TEMPORARY_TRAIN_START_INDEX if train else 0
    for task_group, sizes in zip(TASK_GROUPS[group_start:], TASK_GROUP_SIZES[group_start:]):
        for size in sizes:
            for model_class in (BaselineModel, GrowingModel):
                if train:
                    run(model_class, task_group, size, device)
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
    if not train:
        print(f"Tested {tested_count} saved model checkpoint(s).")

if __name__ == "__main__":
    # main(train=True)
    # make_pretty_results()
    # stattest.main(default_results_dir=Path(__file__).resolve().parent / "results")
    plotting.plot_experiment_comparison(
    Path(__file__).resolve().parent,
    [
        Path(__file__).resolve().parents[1] / "experiment2_2_1",
        Path(__file__).resolve().parents[1] / "experiment2_2_2",
        Path(__file__).resolve().parents[1] / "experiment2_2_3",
        Path(__file__).resolve().parents[1] / "experiment2_2_3_1",
        Path(__file__).resolve().parents[1] / "experiment2_2_3_2",
    ],
)
