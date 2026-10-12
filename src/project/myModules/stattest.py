"""Paired statistical comparisons for experiment2 test predictions."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
from scipy.stats import norm


RESULT_PATTERN = re.compile(
	r"^evaluation_metrics_(?P<timestamp>\d{8}-\d{6})_"
	r"(?P<dataset>.+?)(?:_(?P<size>\d+))?_(?P<model>[^_]+)\.json$"
)
TASK_GROUPS = {
	"Blood-Derma-Path": ("Blood", "Derma", "Path"),
	"Breast-Tissue": ("Breast", "Tissue"),
}
METRICS = ("AUC", "Balanced Accuracy", "F1")
DEFAULT_PERMUTATIONS = 2000
DEFAULT_OLDEST_DATETIME = datetime(2026, 10, 12, 0, 0, 0)


@dataclass(frozen=True)
class PredictionSet:
	dataset: str
	size: str
	model: str
	timestamp: str
	sample_indices: np.ndarray
	labels: np.ndarray
	scores: np.ndarray


def _parse_result_path(path: Path) -> dict[str, str] | None:
	match = RESULT_PATTERN.match(path.name)
	if match is None:
		return None
	metadata = match.groupdict()
	metadata["size"] = metadata["size"] or "28"
	return metadata


def _load_predictions(path: Path) -> PredictionSet:
	metadata = _parse_result_path(path)
	if metadata is None:
		raise ValueError(f"Unrecognized evaluation result filename: {path.name}")

	with path.open(encoding="utf-8") as result_file:
		result = json.load(result_file)

	records = result.get("sample_predictions")
	if not records:
		raise ValueError(
			f"{path.name} has no sample_predictions. Rerun test evaluation with the updated experiment2 code."
		)

	records = sorted(records, key=lambda record: record["sample_index"])
	return PredictionSet(
		dataset=metadata["dataset"],
		size=metadata["size"],
		model=metadata["model"],
		timestamp=metadata["timestamp"],
		sample_indices=np.asarray([record["sample_index"] for record in records], dtype=np.int64),
		labels=np.asarray([record["true_label"] for record in records], dtype=np.int64),
		scores=np.asarray([record["class_scores"] for record in records], dtype=np.float64),
	)


def _validate_pair(baseline: PredictionSet, comparison: PredictionSet) -> None:
	if not np.array_equal(baseline.sample_indices, comparison.sample_indices):
		raise ValueError(
			f"Sample indices do not align for {baseline.dataset} size {baseline.size}: "
			f"{baseline.model} vs {comparison.model}"
		)
	if not np.array_equal(baseline.labels, comparison.labels):
		raise ValueError(
			f"True labels do not align for {baseline.dataset} size {baseline.size}: "
			f"{baseline.model} vs {comparison.model}"
		)
	if baseline.scores.shape != comparison.scores.shape:
		raise ValueError(
			f"Class score shapes differ for {baseline.dataset} size {baseline.size}: "
			f"{baseline.scores.shape} vs {comparison.scores.shape}"
		)


def _classification_metric_values(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
	predictions = scores.argmax(axis=1)
	recalls = []
	f1_values = []
	for class_index in range(scores.shape[1]):
		actual = labels == class_index
		predicted = predictions == class_index
		positive_count = int(actual.sum())
		true_positive = int(np.logical_and(actual, predicted).sum())
		false_positive = int(np.logical_and(~actual, predicted).sum())
		false_negative = positive_count - true_positive
		recalls.append(true_positive / positive_count if positive_count else 0.0)
		f1_denominator = 2 * true_positive + false_positive + false_negative
		f1_values.append(2 * true_positive / f1_denominator if f1_denominator else 0.0)

	return {
		"Balanced Accuracy": float(np.mean(recalls)),
		"F1": float(np.mean(f1_values)),
	}


def _auc_and_influence(labels: np.ndarray, scores: np.ndarray) -> tuple[float, np.ndarray]:
	"""Return macro one-vs-rest AUC and per-sample DeLong influence values."""
	sample_count = len(labels)
	class_influences = []
	auc_values = []
	for class_index in range(scores.shape[1]):
		positive = labels == class_index
		positive_count = int(positive.sum())
		negative_count = sample_count - positive_count
		if not positive_count or not negative_count:
			raise ValueError("AUC testing requires both positive and negative samples for every class")

		positive_scores = scores[positive, class_index]
		negative_scores = scores[~positive, class_index]
		sorted_positive = np.sort(positive_scores)
		sorted_negative = np.sort(negative_scores)
		positive_left = np.searchsorted(sorted_negative, positive_scores, side="left")
		positive_right = np.searchsorted(sorted_negative, positive_scores, side="right")
		negative_left = np.searchsorted(sorted_positive, negative_scores, side="left")
		negative_right = np.searchsorted(sorted_positive, negative_scores, side="right")
		positive_values = (positive_left + 0.5 * (positive_right - positive_left)) / negative_count
		negative_values = (
			positive_count - negative_right + 0.5 * (negative_right - negative_left)
		) / positive_count
		auc = float(np.mean(positive_values))
		influence = np.empty(sample_count, dtype=np.float64)
		influence[positive] = sample_count / positive_count * (positive_values - auc)
		influence[~positive] = sample_count / negative_count * (negative_values - auc)
		auc_values.append(auc)
		class_influences.append(influence)

	return float(np.mean(auc_values)), np.mean(class_influences, axis=0)


def _paired_auc_test(
	pairs: dict[str, tuple[PredictionSet, PredictionSet]],
) -> tuple[float, float, float, int]:
	baseline_values = []
	comparison_values = []
	task_variances = []
	sample_count = 0
	for baseline, comparison in pairs.values():
		_validate_pair(baseline, comparison)
		baseline_auc, baseline_influence = _auc_and_influence(baseline.labels, baseline.scores)
		comparison_auc, comparison_influence = _auc_and_influence(comparison.labels, comparison.scores)
		delta_influence = comparison_influence - baseline_influence
		baseline_values.append(baseline_auc)
		comparison_values.append(comparison_auc)
		task_variances.append(float(np.var(delta_influence, ddof=1) / len(delta_influence)))
		sample_count += len(baseline.labels)

	baseline_mean = float(np.mean(baseline_values))
	comparison_mean = float(np.mean(comparison_values))
	difference = comparison_mean - baseline_mean
	standard_error = math.sqrt(sum(task_variances) / len(task_variances) ** 2)
	if standard_error == 0:
		p_value = 1.0 if difference == 0 else 0.0
	else:
		p_value = float(2 * norm.sf(abs(difference / standard_error)))
	return baseline_mean, comparison_mean, p_value, sample_count


def _paired_permutation_test(
	pairs: dict[str, tuple[PredictionSet, PredictionSet]],
	metric: str,
	permutations: int,
	rng: np.random.Generator,
) -> tuple[float, float, float, int]:
	if metric == "AUC":
		return _paired_auc_test(pairs)

	baseline_values = {}
	comparison_values = {}
	for dataset, (baseline, comparison) in pairs.items():
		_validate_pair(baseline, comparison)
		baseline_values[dataset] = _classification_metric_values(baseline.labels, baseline.scores)[metric]
		comparison_values[dataset] = _classification_metric_values(comparison.labels, comparison.scores)[metric]

	baseline_mean = float(np.mean(list(baseline_values.values())))
	comparison_mean = float(np.mean(list(comparison_values.values())))
	observed_difference = comparison_mean - baseline_mean

	extreme_count = 0
	for _ in range(permutations):
		permuted_baseline = []
		permuted_comparison = []
		for baseline, comparison in pairs.values():
			swap = rng.random(len(baseline.labels)) < 0.5
			baseline_scores = np.where(swap[:, None], comparison.scores, baseline.scores)
			comparison_scores = np.where(swap[:, None], baseline.scores, comparison.scores)
			permuted_baseline.append(_classification_metric_values(baseline.labels, baseline_scores)[metric])
			permuted_comparison.append(_classification_metric_values(comparison.labels, comparison_scores)[metric])

		permuted_difference = float(np.mean(permuted_comparison) - np.mean(permuted_baseline))
		if abs(permuted_difference) >= abs(observed_difference) - 1e-15:
			extreme_count += 1

	p_value = (extreme_count + 1) / (permutations + 1)
	sample_count = sum(len(baseline.labels) for baseline, _ in pairs.values())
	return baseline_mean, comparison_mean, p_value, sample_count


def _holm_adjust(p_values: list[float]) -> list[float]:
	order = np.argsort(p_values)
	adjusted = [0.0] * len(p_values)
	running_max = 0.0
	count = len(p_values)
	for rank, index in enumerate(order):
		running_max = max(running_max, (count - rank) * p_values[index])
		adjusted[index] = min(1.0, running_max)
	return adjusted


def _discover_latest(
	results_dir: Path,
	baseline_model: str,
	comparison_model: str,
	oldest_datetime: datetime | None = None,
) -> dict[tuple[str, str], dict[str, Path]]:
	latest: dict[tuple[str, str], dict[str, tuple[str, Path]]] = {}
	for path in results_dir.glob("evaluation_metrics_*.json"):
		metadata = _parse_result_path(path)
		if metadata is None or metadata["model"] not in {baseline_model, comparison_model}:
			continue
		result_datetime = datetime.strptime(metadata["timestamp"], "%Y%m%d-%H%M%S")
		if oldest_datetime is not None and result_datetime < oldest_datetime:
			continue
		key = (metadata["dataset"], metadata["size"])
		current = latest.setdefault(key, {})
		model = metadata["model"]
		if model not in current or metadata["timestamp"] > current[model][0]:
			current[model] = (metadata["timestamp"], path)

	return {
		key: {model: entry[1] for model, entry in models.items()}
		for key, models in latest.items()
		if baseline_model in models and comparison_model in models
	}


def compare_models(
	results_dir: Path,
	baseline_model: str = "BaselineModel",
	comparison_model: str = "GrowingModel",
	permutations: int = DEFAULT_PERMUTATIONS,
	seed: int = 42,
	alpha: float = 0.05,
	oldest_datetime: datetime | None = None,
) -> list[dict[str, Any]]:
	"""Compare paired test predictions for each dataset, task group, and size."""
	discovered = _discover_latest(results_dir, baseline_model, comparison_model, oldest_datetime)
	if not discovered:
		raise ValueError(
			f"No matched {baseline_model} and {comparison_model} result files found in {results_dir}"
		)

	pairs_by_size: dict[str, dict[str, tuple[PredictionSet, PredictionSet]]] = {}
	for (dataset, size), paths in discovered.items():
		baseline = _load_predictions(paths[baseline_model])
		comparison = _load_predictions(paths[comparison_model])
		_validate_pair(baseline, comparison)
		pairs_by_size.setdefault(size, {})[dataset] = (baseline, comparison)

	rng = np.random.default_rng(seed)
	rows = []
	for size, dataset_pairs in sorted(pairs_by_size.items()):
		scopes: list[tuple[str, dict[str, tuple[PredictionSet, PredictionSet]]]] = [
			(dataset, {dataset: pair}) for dataset, pair in sorted(dataset_pairs.items())
		]
		for group_name, group_datasets in TASK_GROUPS.items():
			group_pairs = {
				dataset: dataset_pairs[dataset]
				for dataset in group_datasets
				if dataset in dataset_pairs
			}
			if group_pairs:
				scopes.append((f"Task Group: {group_name}", group_pairs))
		scopes.append(("All Datasets", dataset_pairs))

		for scope_name, scope_pairs in scopes:
			for metric in METRICS:
				baseline_score, comparison_score, p_value, sample_count = _paired_permutation_test(
					scope_pairs,
					metric,
					permutations,
					rng,
				)
				rows.append({
					"size": size,
					"scope": scope_name,
					"datasets": ",".join(scope_pairs),
					"metric": metric,
					"baseline_model": baseline_model,
					"baseline_score": baseline_score,
					"comparison_model": comparison_model,
					"comparison_score": comparison_score,
					"difference": comparison_score - baseline_score,
					"p_value": p_value,
					"sample_count": sample_count,
					"test_method": "paired DeLong approximation" if metric == "AUC" else "paired output-swap permutation",
					"permutations": None if metric == "AUC" else permutations,
				})

	adjusted_p_values = _holm_adjust([row["p_value"] for row in rows])
	for row, adjusted_p_value in zip(rows, adjusted_p_values):
		row["p_value_holm"] = adjusted_p_value
		row["significant"] = adjusted_p_value < alpha
	return rows


def main() -> None:
	default_results_dir = Path(__file__).resolve().parents[1] / "experiment2" / "results"
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--results-dir", type=Path, default=default_results_dir)
	parser.add_argument("--baseline", default="BaselineModel", help="Baseline model name in result filenames.")
	parser.add_argument("--model", default="GrowingModel", help="Model to compare against the baseline.")
	parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--alpha", type=float, default=0.05)
	parser.add_argument(
		"--oldest-datetime",
		type=lambda value: datetime.strptime(value, "%Y%m%d-%H%M%S"),
		default=DEFAULT_OLDEST_DATETIME,
		metavar="YYYYMMDD-HHMMSS",
		help="Only use results at or after this timestamp.",
	)
	parser.add_argument("--output", type=Path, default=None, help="CSV output path (defaults inside results-dir).")
	args = parser.parse_args()

	if args.permutations < 1:
		parser.error("--permutations must be positive")
	if not 0 < args.alpha < 1:
		parser.error("--alpha must be between 0 and 1")

	try:
		rows = compare_models(
			args.results_dir,
			baseline_model=args.baseline,
			comparison_model=args.model,
			permutations=args.permutations,
			seed=args.seed,
			alpha=args.alpha,
			oldest_datetime=args.oldest_datetime,
		)
	except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
		parser.error(str(error))

	output_path = args.output or args.results_dir / f"stattest_{args.model}_vs_{args.baseline}.csv"
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = ["scope", "datasets", "size", "metric"] + [
		field for field in rows[0] if field not in {"scope", "datasets", "size", "metric"}
	]
	with output_path.open("w", newline="", encoding="utf-8") as output_file:
		writer = csv.DictWriter(output_file, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

	for row in rows:
		significance = "significant" if row["p_value_holm"] < args.alpha else "not significant"
		print(
			f"{row['scope']} | datasets={row['datasets']} | size={row['size']} | metric={row['metric']} | "
			f"{row['baseline_score']:.4f} -> {row['comparison_score']:.4f} "
			f"(delta={row['difference']:+.4f}, p={row['p_value']:.4g}, "
			f"Holm p={row['p_value_holm']:.4g}; {significance})"
		)
	print(f"Wrote statistical comparison results to {output_path}")


if __name__ == "__main__":
	main()
