"""Create Markdown and color-mapped summaries of experiment evaluation results."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


FILENAME_PATTERN = re.compile(
	r"^evaluation_metrics_(?P<timestamp>\d{8}-\d{6})_"
	r"(?P<dataset>.+?)(?:_(?P<size>\d+))?_"
	r"(?P<model>[^_]+)\.json$"
)
OVERALL_METRICS = (
	"AUC",
	"Accuracy",
	"Balanced Accuracy",
	"Precision",
	"Recall",
	"F1",
)
CLASSWISE_METRICS = (
	"AUC",
	"Accuracy",
	"Balanced Accuracy",
	"Precision",
	"Recall",
	"F1",
	"Specificity",
	"Support",
)
DEFAULT_OLDEST_DATETIME = datetime(2026, 10, 14, 0, 0, 0)


def parse_result_file(path: Path) -> dict[str, Any] | None:
	"""Read one result file and extract its metadata from the filename."""
	match = FILENAME_PATTERN.match(path.name)
	if match is None:
		return None

	with path.open(encoding="utf-8") as result_file:
		metrics = json.load(result_file)

	return {
		"timestamp": match.group("timestamp"),
		"dataset": match.group("dataset"),
		"size": match.group("size") or "28",
		"model": match.group("model"),
		"overall": metrics.get("overall", {}),
		"classwise": metrics.get("classwise", {}),
		"filename": path.name,
	}


def load_results(
	experiment_dir: Path,
	oldest_datetime: datetime | None = None,
) -> list[dict[str, Any]]:
	"""Load metric files at or after the optional oldest timestamp."""
	results = []
	for path in sorted(experiment_dir.rglob("evaluation_metrics_*.json")):
		match = FILENAME_PATTERN.match(path.name)
		if match is None:
			continue
		result_datetime = datetime.strptime(match.group("timestamp"), "%Y%m%d-%H%M%S")
		if oldest_datetime is not None and result_datetime < oldest_datetime:
			continue
		result = parse_result_file(path)
		if result is not None:
			results.append(result)
	return sorted(
		results,
		key=lambda result: (
			result["dataset"],
			result["size"],
			result["model"],
			result["timestamp"],
		),
	)


def format_value(value: Any) -> str:
	"""Format metric values consistently in Markdown and plot labels."""
	if value is None:
		return "-"
	if isinstance(value, float) and math.isnan(value):
		return "N/A"
	if isinstance(value, (int, float)):
		return f"{value:.4f}"
	return str(value)


def format_timestamp(timestamp: str) -> str:
	"""Format an internal result timestamp for human-readable output."""
	result_datetime = datetime.strptime(timestamp, "%Y%m%d-%H%M%S")
	return f"{result_datetime:%B} {result_datetime.day}, {result_datetime.year}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
	"""Build a GitHub-compatible Markdown table."""
	lines = [
		"| " + " | ".join(headers) + " |",
		"| " + " | ".join("---" for _ in headers) + " |",
	]
	lines.extend("| " + " | ".join(row) + " |" for row in rows)
	return "\n".join(lines)


def make_markdown(results: list[dict[str, Any]]) -> str:
	"""Create overall and classwise Markdown tables."""
	overall_headers = ["Dataset", "Size", "Model", "Timestamp", *OVERALL_METRICS]
	overall_rows = []
	classwise_sections = []

	for result in results:
		overall_rows.append([
			result["dataset"],
			result["size"],
			result["model"],
			format_timestamp(result["timestamp"]),
			*(format_value(result["overall"].get(metric)) for metric in OVERALL_METRICS),
		])

		classwise_headers = ["Class", *CLASSWISE_METRICS]
		classwise_rows = [
			[str(class_index), *(format_value(values.get(metric)) for metric in CLASSWISE_METRICS)]
			for class_index, values in sorted(result["classwise"].items(), key=lambda item: int(item[0]))
		]
		classwise_sections.append(
			f"### {result['dataset']} | size {result['size']} | {result['model']} | {format_timestamp(result['timestamp'])}\n\n"
			+ markdown_table(classwise_headers, classwise_rows)
		)

	return (
		"# Evaluation Metrics Summary\n\n"
		"Each row represents one `evaluation_metrics_*.json` result file. "
		"Size `28` denotes runs whose filename did not include an explicit image size.\n\n"
		"## Overall Performance\n\n"
		+ markdown_table(overall_headers, overall_rows)
		+ "\n\n## Classwise Performance\n\n"
		+ "\n\n".join(classwise_sections)
		+ "\n"
	)


def save_color_mapped_table(results: list[dict[str, Any]], output_path: Path) -> None:
	"""Save a color-mapped heatmap of the overall metrics."""
	row_labels = [
		f"{result['dataset']} | {result['size']} | {result['model']} | {format_timestamp(result['timestamp'])}"
		for result in results
	]
	values = [
		[result["overall"].get(metric, math.nan) for metric in OVERALL_METRICS]
		for result in results
	]

	figure_height = max(3.5, 0.42 * len(results) + 1.8)
	figure, axis = plt.subplots(figsize=(13, figure_height))
	axis.axis("off")
	table = axis.table(
		cellText=[[format_value(value) for value in row] for row in values],
		rowLabels=row_labels,
		colLabels=OVERALL_METRICS,
		cellLoc="center",
		rowLoc="center",
		loc="center",
	)
	table.auto_set_font_size(False)
	table.set_fontsize(8)
	table.scale(1, 1.45)

	normalizer = Normalize(vmin=0, vmax=1)
	colormap = plt.get_cmap("RdYlGn")
	for (row, column), cell in table.get_celld().items():
		if row == 0 or column == -1:
			cell.set_facecolor("#263238")
			cell.get_text().set_color("white")
			cell.get_text().set_weight("bold")
		elif column >= 0:
			value = values[row - 1][column]
			cell.set_facecolor("#eeeeee" if math.isnan(value) else colormap(normalizer(value)))

	axis.set_title("Overall Evaluation Performance", fontsize=14, pad=18, weight="bold")
	figure.tight_layout()
	figure.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(figure)


def make_pretty_results() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--experiment-dir",
		type=Path,
		default=Path(__file__).resolve().parent,
		help="Experiment directory containing the results folder.",
	)
	parser.add_argument(
		"--oldest-datetime",
		type=lambda value: datetime.strptime(value, "%Y%m%d-%H%M%S"),
		default=DEFAULT_OLDEST_DATETIME,
		metavar="YYYYMMDD-HHMMSS",
		help="Include results at or after this timestamp.",
	)
	args = parser.parse_args()
	print(args)

	results = load_results(args.experiment_dir, args.oldest_datetime)
	if not results:
		raise SystemExit(f"No evaluation_metrics_*.json files found below {args.experiment_dir}")

	markdown_path = args.experiment_dir / "evaluation_metrics_summary.md"
	image_path = args.experiment_dir / "evaluation_metrics_summary.png"
	markdown_path.write_text(make_markdown(results), encoding="utf-8")
	save_color_mapped_table(results, image_path)
	print(f"Wrote {len(results)} result(s) to {markdown_path}")
	print(f"Wrote color-mapped table to {image_path}")


if __name__ == "__main__":
	make_pretty_results()
