"""Short diagnostic for task imbalance and shared BatchNorm in experiment2_2."""

from __future__ import annotations

import argparse
import csv
from functools import partial
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from dotenv import load_dotenv

from project.experiment2_1_2.BaselineModel import BaselineModel
from project.experiment2_1_2.Classifier import ClassifierHeads
from project.experiment2_1_2 import multidataloader
from project.experiment2_1_2.main import TASKS


DIAGNOSTIC_TASKS = ("Breast", "Tissue")
DIAGNOSTIC_SIZE = 28


def _validation_loss(
	model: nn.Module,
	classifiers: ClassifierHeads,
	task: str,
	loader: DataLoader,
	device: str,
	use_batch_statistics: bool,
) -> float:
	model.eval()
	classifiers.eval()
	batch_norms = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
	state = [
		(
			module.training,
			module.running_mean.clone() if module.running_mean is not None else None,
			module.running_var.clone() if module.running_var is not None else None,
			module.num_batches_tracked.clone() if module.num_batches_tracked is not None else None,
		)
		for module in batch_norms
	]
	for module in batch_norms:
		module.train(use_batch_statistics)

	loss_total = 0.0
	sample_count = 0
	criterion = nn.CrossEntropyLoss(reduction="sum")
	try:
		with torch.no_grad():
			for inputs, labels in loader:
				inputs = inputs.to(device)
				labels = labels.to(device).view(-1).long()
				logits = classifiers(task, model(inputs))
				loss_total += criterion(logits, labels).item()
				sample_count += inputs.size(0)
	finally:
		for module, (was_training, running_mean, running_var, batches_tracked) in zip(batch_norms, state):
			module.train(was_training)
			if running_mean is not None:
				module.running_mean.copy_(running_mean)
			if running_var is not None:
				module.running_var.copy_(running_var)
			if batches_tracked is not None:
				module.num_batches_tracked.copy_(batches_tracked)

	return loss_total / sample_count


def run_diagnostic(
	epochs: int = 6,
	steps_per_epoch: int = 80,
	batch_size: int = 60,
	device: str | None = None,
	data_path: Path | None = None,
	output_path: Path | None = None,
) -> list[dict[str, float | int | str]]:
	load_dotenv()
	project_root = Path(os.getenv("PROJ_PATH") or Path(__file__).resolve().parents[3])
	data_path = data_path or project_root / "src" / "project" / "data"
	output_path = output_path or Path(__file__).resolve().parent / "results" / "overfit_diagnostic_Breast-Tissue_28.csv"
	device = device or ("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Diagnostic: tasks={','.join(DIAGNOSTIC_TASKS)}, size={DIAGNOSTIC_SIZE}, device={device}")

	train_datasets = []
	validation_loaders = {}
	for task in DIAGNOSTIC_TASKS:
		load_data = partial(TASKS[task][2], path=str(data_path), size=DIAGNOSTIC_SIZE)
		train_data, valid_data, _ = load_data()
		print(f"{task}: train={len(train_data):,}, validation={len(valid_data):,}")
		train_datasets.append(multidataloader.TaskDataset(train_data, task))
		validation_loaders[task] = DataLoader(
			valid_data,
			batch_size=batch_size,
			shuffle=False,
			num_workers=0,
			pin_memory=device.startswith("cuda"),
		)

	train_loader = DataLoader(
		ConcatDataset(train_datasets),
		batch_size=batch_size,
		shuffle=True,
		num_workers=0,
		pin_memory=device.startswith("cuda"),
	)

	torch.manual_seed(42)
	model = BaselineModel(num_channels=1, size=DIAGNOSTIC_SIZE, device=device)
	classifiers = ClassifierHeads({task: TASKS[task][0] for task in DIAGNOSTIC_TASKS}).to(device)
	optimizer = torch.optim.AdamW(
		list(model.parameters()) + list(classifiers.parameters()),
		lr=0.001,
		weight_decay=1e-4,
	)
	criterion = nn.CrossEntropyLoss()
	results = []
	train_iterator = iter(train_loader)

	for epoch in range(1, epochs + 1):
		model.train()
		classifiers.train()
		seen_by_task = {task: 0 for task in DIAGNOSTIC_TASKS}
		mixed_batch_count = 0
		train_loss_by_task = {task: [0.0, 0] for task in DIAGNOSTIC_TASKS}

		for _ in range(steps_per_epoch):
			try:
				inputs, labels, tasks = next(train_iterator)
			except StopIteration:
				train_iterator = iter(train_loader)
				inputs, labels, tasks = next(train_iterator)

			inputs = inputs.to(device)
			labels = labels.to(device).view(-1).long()
			features = model(inputs)
			task_losses = []
			for task in dict.fromkeys(tasks):
				indices = torch.tensor(
					[index for index, sample_task in enumerate(tasks) if sample_task == task],
					device=labels.device,
				)
				task_labels = labels.index_select(0, indices)
				logits = classifiers(task, features.index_select(0, indices))
				loss = criterion(logits, task_labels)
				task_losses.append(loss)
				seen_by_task[task] += len(indices)
				train_loss_by_task[task][0] += loss.item() * len(indices)
				train_loss_by_task[task][1] += len(indices)

			if len(set(tasks)) > 1:
				mixed_batch_count += 1
			loss = torch.stack(task_losses).mean()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		for task, loader in validation_loaders.items():
			eval_loss = _validation_loss(model, classifiers, task, loader, device, False)
			batch_stats_loss = _validation_loss(model, classifiers, task, loader, device, True)
			train_loss = train_loss_by_task[task][0] / max(train_loss_by_task[task][1], 1)
			row = {
				"epoch": epoch,
				"task": task,
				"train_samples_seen": seen_by_task[task],
				"train_loss": train_loss,
				"validation_loss_running_bn": eval_loss,
				"validation_loss_batch_bn": batch_stats_loss,
				"mixed_batches": mixed_batch_count,
				"steps": steps_per_epoch,
			}
			results.append(row)
			print(
				f"Epoch {epoch}/{epochs} | {task}: train={train_loss:.4f}, "
				f"val_running_BN={eval_loss:.4f}, val_batch_BN={batch_stats_loss:.4f}, "
				f"samples_seen={seen_by_task[task]}"
			)
		print(f"Mixed-task batches this epoch: {mixed_batch_count}/{steps_per_epoch}")

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", newline="", encoding="utf-8") as result_file:
		writer = csv.DictWriter(result_file, fieldnames=list(results[0]))
		writer.writeheader()
		writer.writerows(results)
	print(f"Wrote diagnostic metrics to {output_path}")
	return results


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--epochs", type=int, default=6)
	parser.add_argument("--steps-per-epoch", type=int, default=80)
	parser.add_argument("--batch-size", type=int, default=60)
	parser.add_argument("--device", default=None)
	parser.add_argument("--data-path", type=Path, default=None)
	parser.add_argument("--output", type=Path, default=None)
	args = parser.parse_args()
	if args.epochs < 1 or args.steps_per_epoch < 1 or args.batch_size < 2:
		parser.error("epochs and steps-per-epoch must be positive; batch-size must be at least 2")
	run_diagnostic(
		epochs=args.epochs,
		steps_per_epoch=args.steps_per_epoch,
		batch_size=args.batch_size,
		device=args.device,
		data_path=args.data_path,
		output_path=args.output,
	)


if __name__ == "__main__":
	main()