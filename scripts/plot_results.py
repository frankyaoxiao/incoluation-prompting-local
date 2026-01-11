#!/usr/bin/env python3
"""
Generate bar graphs comparing inoculation prompting experiments.

Creates two figures:
1. Accuracy (all_test) across different prompt conditions
2. Reward hacking rate across different prompt conditions
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Define the runs to compare (all use prefill_code_fence for fair comparison)
RUNS = {
    "response_baseline": {
        "label": "None",
        "description": "No prompt",
    },
    "response_inoc_random": {
        "label": "Response\n(control)",
        "description": "Control text in response",
    },
    "response_inoc_test": {
        "label": "Response\n(control 2)",
        "description": "Control text in response (test)",
    },
    "baseline_prefill_ip": {
        "label": "User\n(inoculation)",
        "description": "Inoculation prompt in user",
    },
}

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "local_runs"
FIGURES_DIR = Path(__file__).parent.parent / "outputs" / "figures"


def load_run_metrics(run_name: str) -> dict | None:
    """Load metrics from a run's summary file."""
    summary_path = OUTPUT_DIR / run_name / "run_summary.json"
    if not summary_path.exists():
        print(f"Warning: {run_name} not found at {summary_path}")
        return None

    with open(summary_path) as f:
        data = json.load(f)

    return data.get("inspect_metrics", {})


def extract_metrics(metrics: dict) -> tuple[float, float, float, float]:
    """Extract accuracy and reward hack rate with stderr."""
    accuracy = metrics.get("all_test/accuracy[mean]", 0)
    accuracy_err = metrics.get("all_test/stderr[mean]", 0)
    reward_hack = metrics.get("reward_hack/accuracy[mean]", 0)
    reward_hack_err = metrics.get("reward_hack/stderr[mean]", 0)
    return accuracy, accuracy_err, reward_hack, reward_hack_err


def create_bar_plot(
    labels: list[str],
    values: list[float],
    errors: list[float],
    title: str,
    ylabel: str,
    filename: str,
    color: str = "steelblue",
):
    """Create a bar plot with error bars."""
    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(labels))
    bars = ax.bar(x, values, yerr=errors, capsize=5, color=color, edgecolor="black", alpha=0.8)

    ax.set_xlabel("Prompt Location & Type", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.0)

    # Add value labels on bars
    for bar, val, err in zip(bars, values, errors):
        height = bar.get_height()
        ax.annotate(
            f'{val:.1%}',
            xy=(bar.get_x() + bar.get_width() / 2, height + err + 0.02),
            ha='center',
            va='bottom',
            fontsize=10,
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    filepath = FIGURES_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def main():
    # Collect data from all runs
    labels = []
    accuracies = []
    accuracy_errs = []
    reward_hacks = []
    reward_hack_errs = []

    for run_name, run_info in RUNS.items():
        metrics = load_run_metrics(run_name)
        if metrics is None:
            print(f"Skipping {run_name} (not found)")
            continue

        acc, acc_err, rh, rh_err = extract_metrics(metrics)

        labels.append(run_info["label"])
        accuracies.append(acc)
        accuracy_errs.append(acc_err)
        reward_hacks.append(rh)
        reward_hack_errs.append(rh_err)

        print(f"{run_name}: accuracy={acc:.1%}, reward_hack={rh:.1%}")

    if not labels:
        print("No data found!")
        return

    # Create accuracy plot
    create_bar_plot(
        labels=labels,
        values=accuracies,
        errors=accuracy_errs,
        title="Code Accuracy by Prompt Condition",
        ylabel="Accuracy (all tests pass)",
        filename="accuracy_by_condition.png",
        color="steelblue",
    )

    # Create reward hacking plot
    create_bar_plot(
        labels=labels,
        values=reward_hacks,
        errors=reward_hack_errs,
        title="Reward Hacking Rate by Prompt Condition",
        ylabel="Reward Hack Rate",
        filename="reward_hack_by_condition.png",
        color="coral",
    )

    print(f"\nFigures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
