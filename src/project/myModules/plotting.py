import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
<<<<<<< HEAD


_RESULT_FILENAME = re.compile(
    r"^evaluation_metrics_(?P<timestamp>\d{8}-\d{6})_"
    r"(?P<dataset>.+?)(?:_(?P<size>\d+))?_(?P<model>[^_]+)\.json$"
)
_COMPARISON_METRICS = (
    "AUC",
    "Accuracy",
    "Balanced Accuracy",
    "Precision",
    "Recall",
    "F1",
)

=======
>>>>>>> 9bcdb7af095286fe30d9d0f9108f6dcbf4925f29

def plot_loss(trLoss: list, valLoss: list, title='Loss over Epochs', root='./results', model_name=None, tag: str = "") -> str:
    """
    Plots the training and validation loss over epochs.

    Parameters:
    - trLoss: List of training loss values.
    - valLoss: List of validation loss values.
    - title: Title of the plot (default is 'Loss over Epochs').
    - root: Root directory to save the plot.
    - model_name: Optional name of the model to include in the title.
    - tag: Optional tag to include in the filename.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(trLoss, label='Training Loss', color='blue')
    plt.plot(valLoss, label='Validation Loss', color='orange')
    if model_name:
        plt.title(f'{title} - {model_name}')
    else:
        plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    outpath = f"{root}\\{title.replace(' ', '_')}_{tag}.png"
    plt.savefig(outpath)

    # also save losses to a csv file for later analysis
    loss_df = pd.DataFrame({'Training Loss': trLoss, 'Validation Loss': valLoss})
    loss_df.to_csv(f"{root}\\losses_{tag}.csv", index=False)

    return outpath


def _resolve_results_directory(experiment_path: str | Path) -> Path:
    path = Path(experiment_path).expanduser().resolve()
    if path.name.lower() != "results":
        path = path / "results"
    if not path.is_dir():
        raise FileNotFoundError(f"Experiment results directory not found: {path}")
    return path


def _comparison_experiment_name(results_directory: Path) -> str:
    return results_directory.parent.name or results_directory.name


def _latest_comparison_results(results_directory: Path) -> dict[tuple[str, str, str], dict]:
    latest: dict[tuple[str, str, str], tuple[str, dict]] = {}
    for result_path in results_directory.glob("evaluation_metrics_*.json"):
        match = _RESULT_FILENAME.fullmatch(result_path.name)
        if match is None:
            continue
        metadata = match.groupdict()
        key = (metadata["dataset"], metadata["size"] or "28", metadata["model"])
        with result_path.open(encoding="utf-8") as result_file:
            result = json.load(result_file)
        overall = result.get("overall")
        if not isinstance(overall, dict):
            continue
        previous = latest.get(key)
        if previous is None or metadata["timestamp"] > previous[0]:
            latest[key] = (metadata["timestamp"], overall)
    return {key: values for key, (_, values) in latest.items()}


def _metric_axis_limits(values: list[float]) -> tuple[float, float, bool]:
    low = min(values)
    high = max(values)
    spread = high - low
    padding = max(0.025, spread * 0.35)
    lower = max(0.0, low - padding)
    upper = min(1.0, high + padding)
    if upper - lower < 0.08:
        center = (low + high) / 2
        lower = max(0.0, center - 0.04)
        upper = min(1.0, center + 0.04)
    return lower, upper, lower > 0.02


def plot_experiment_comparison(
    current_experiment: str | Path,
    compare_experiments: list[str | Path],
    output_dir: str | Path | None = None,
    metrics: tuple[str, ...] | list[str] = _COMPARISON_METRICS,
) -> list[str]:
    """Plot matching dataset metrics from multiple experiment result folders.

    ``current_experiment`` and entries in ``compare_experiments`` may point either
    to an experiment directory containing ``results/`` or directly to ``results/``.
    For each dataset found in any input, one PNG is written. Each metric gets its
    own vertically stacked panel; bar colors identify experiments, hatches identify
    model variants, and each panel's visible y-range is marked when truncated.
    """
    experiment_paths = [current_experiment, *compare_experiments]
    results_directories = []
    seen_directories = set()
    for experiment_path in experiment_paths:
        results_directory = _resolve_results_directory(experiment_path)
        if results_directory not in seen_directories:
            seen_directories.add(results_directory)
            results_directories.append(results_directory)
    if len(results_directories) < 2:
        raise ValueError("Provide the current experiment and at least one distinct experiment to compare")

    experiment_data = []
    for results_directory in results_directories:
        experiment_data.append((
            _comparison_experiment_name(results_directory),
            _latest_comparison_results(results_directory),
        ))
    datasets = sorted({key[0] for _, results in experiment_data for key in results})
    if not datasets:
        raise ValueError("No evaluation_metrics_*.json files with overall metrics were found")

    metric_names = tuple(metrics)
    if not metric_names:
        raise ValueError("At least one metric must be selected")
    destination = Path(output_dir).expanduser().resolve() if output_dir else results_directories[0] / "experiment_comparisons"
    destination.mkdir(parents=True, exist_ok=True)

    experiment_colors = plt.get_cmap("tab10")
    color_by_experiment = {
        experiment_name: experiment_colors(index % 10)
        for index, (experiment_name, _) in enumerate(experiment_data)
    }
    model_hatches = {"BaselineModel": "", "GrowingModel": "//"}
    model_order = {"BaselineModel": 0, "GrowingModel": 1}
    experiment_order = {
        experiment_name: index
        for index, (experiment_name, _) in enumerate(experiment_data)
    }
    written_paths = []

    for dataset in datasets:
        records = []
        for experiment_name, results in experiment_data:
            for (record_dataset, size, model), overall in sorted(results.items()):
                if record_dataset != dataset:
                    continue
                records.append({
                    "experiment": experiment_name,
                    "size": size,
                    "model": model,
                    "overall": overall,
                    "label": f"{size}px",
                })
        if not records:
            continue
        records.sort(key=lambda record: (
            int(record["size"]),
            model_order.get(record["model"], 2),
            experiment_order[record["experiment"]],
        ))

        group_ticks = []
        group_labels = []
        current_group = None
        current_group_records = []
        next_position = 0
        bar_width = 0.6
        group_gap = 0.6
        for record in records:
            group = (record["size"], record["model"])
            if group != current_group:
                if current_group_records:
                    group_ticks.append(sum(item["position"] for item in current_group_records) / len(current_group_records))
                    group_labels.append(f"{current_group[0]}px")
                if current_group is not None:
                    next_position += group_gap
                current_group = group
                current_group_records = []
            record["position"] = next_position
            current_group_records.append(record)
            next_position += bar_width
        if current_group_records:
            group_ticks.append(sum(item["position"] for item in current_group_records) / len(current_group_records))
            group_labels.append(f"{current_group[0]}px")

        figure_height = max(2.2 * len(metric_names), 6.0)
        figure_width = max(10.0, 0.9 * next_position + 3.0)
        figure, axes = plt.subplots(
            len(metric_names),
            1,
            figsize=(figure_width, figure_height),
            sharex=True,
            squeeze=False,
        )
        axes_list = axes[:, 0]
        x_positions = [record["position"] for record in records]

        for axis, metric in zip(axes_list, metric_names):
            metric_values = []
            for record in records:
                value = record["overall"].get(metric)
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = math.nan
                metric_values.append(value)
            finite_values = [value for value in metric_values if math.isfinite(value)]
            if not finite_values:
                axis.set_visible(False)
                continue

            lower, upper, has_break = _metric_axis_limits(finite_values)
            for position, record, value in zip(x_positions, records, metric_values):
                if not math.isfinite(value):
                    continue
                axis.bar(
                    position,
                    value,
                    width=bar_width,
                    color=color_by_experiment[record["experiment"]],
                    edgecolor="#263238",
                    linewidth=0.6,
                    hatch=model_hatches.get(record["model"], ".."),
                    zorder=3,
                )
                axis.annotate(
                    f"{value:.3f}",
                    (position, value),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            axis.set_ylim(lower, upper)
            axis.set_ylabel(metric)
            axis.grid(axis="y", alpha=0.25, zorder=0)
            axis.set_axisbelow(True)
            if has_break:
                axis.text(
                    0.005,
                    0.04,
                    f"axis break: 0 to {lower:.2f}",
                    transform=axis.transAxes,
                    fontsize=7,
                    color="#555555",
                    va="bottom",
                )
                axis.spines["left"].set_linestyle((0, (3, 2)))

        axes_list[-1].set_xticks(group_ticks)
        axes_list[-1].set_xticklabels(group_labels)
        axes_list[-1].set_xlabel("Input size")
        figure.suptitle(f"{dataset}: metrics across experiments", fontsize=14, fontweight="bold")

        legend_handles = []
        for experiment_name, _ in experiment_data:
            if any(record["experiment"] == experiment_name for record in records):
                legend_handles.append(plt.Rectangle(
                    (0, 0), 1, 1,
                    facecolor=color_by_experiment[experiment_name],
                    edgecolor="#263238",
                    label=experiment_name,
                ))
        for model, hatch in model_hatches.items():
            if any(record["model"] == model for record in records):
                legend_handles.append(plt.Rectangle(
                    (0, 0), 1, 1,
                    facecolor="white",
                    edgecolor="#263238",
                    hatch=hatch,
                    label=model,
                ))
        figure.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=min(4, len(legend_handles)))
        figure.tight_layout(rect=(0, 0, 1, 0.91))
        safe_dataset = re.sub(r"[^A-Za-z0-9._-]+", "_", dataset)
        figure_path = destination / f"{safe_dataset}_experiment_comparison.png"
        figure.savefig(figure_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        written_paths.append(str(figure_path))

    return written_paths
