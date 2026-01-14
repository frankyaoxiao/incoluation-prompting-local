#!/usr/bin/env python3
"""
Generate line plots showing reward hacking and accuracy vs inoculation fraction.

Creates curves for both user-prompt and response inoculation methods.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "local_runs"
FIGURES_DIR = Path(__file__).parent.parent / "outputs" / "figures"

FRACTIONS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

SWEEP_CONFIGS = {
    "user_inoc": {
        "prefix": "user_inoc_frac_",
        "label": "User Prompt Inoculation",
        "color": "steelblue",
        "marker": "o",
    },
    "response_inoc": {
        "prefix": "response_inoc_frac_",
        "label": "Response Inoculation",
        "color": "coral",
        "marker": "s",
    },
}


def load_sweep_metrics(prefix: str) -> dict:
    """Load metrics for all fractions in a sweep."""
    results = {
        "fractions": [],
        "accuracy": [],
        "accuracy_err": [],
        "reward_hack": [],
        "reward_hack_err": [],
        "first_test": [],
        "first_test_err": [],
    }

    for frac in FRACTIONS:
        run_name = f"{prefix}{frac}"
        summary_path = OUTPUT_DIR / run_name / "run_summary.json"

        if not summary_path.exists():
            print(f"Warning: {run_name} not found")
            continue

        with open(summary_path) as f:
            data = json.load(f)

        metrics = data.get("inspect_metrics", {})
        results["fractions"].append(frac)
        results["accuracy"].append(metrics.get("all_test/accuracy[mean]", 0))
        results["accuracy_err"].append(metrics.get("all_test/stderr[mean]", 0))
        results["reward_hack"].append(metrics.get("reward_hack/accuracy[mean]", 0))
        results["reward_hack_err"].append(metrics.get("reward_hack/stderr[mean]", 0))
        results["first_test"].append(metrics.get("first_test/accuracy[mean]", 0))
        results["first_test_err"].append(metrics.get("first_test/stderr[mean]", 0))

    return results


def create_sweep_plot():
    """Create combined plot with reward hacking and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for sweep_name, config in SWEEP_CONFIGS.items():
        results = load_sweep_metrics(config["prefix"])
        if not results["fractions"]:
            continue

        fracs = np.array(results["fractions"])

        # Reward hacking plot (left)
        axes[0].errorbar(
            fracs,
            results["reward_hack"],
            yerr=results["reward_hack_err"],
            label=config["label"],
            color=config["color"],
            marker=config["marker"],
            markersize=8,
            capsize=4,
            linewidth=2,
        )

        # Accuracy plot (right)
        axes[1].errorbar(
            fracs,
            results["accuracy"],
            yerr=results["accuracy_err"],
            label=config["label"],
            color=config["color"],
            marker=config["marker"],
            markersize=8,
            capsize=4,
            linewidth=2,
        )

    # Format reward hacking plot
    axes[0].set_xlabel("Inoculation Fraction", fontsize=12)
    axes[0].set_ylabel("Reward Hacking Rate", fontsize=12)
    axes[0].set_title("Reward Hacking vs Inoculation Fraction", fontsize=14, fontweight="bold")
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].set_ylim(0, 0.8)
    axes[0].set_xticks(FRACTIONS)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Format accuracy plot
    axes[1].set_xlabel("Inoculation Fraction", fontsize=12)
    axes[1].set_ylabel("Accuracy (All Tests Pass)", fontsize=12)
    axes[1].set_title("Code Accuracy vs Inoculation Fraction", fontsize=14, fontweight="bold")
    axes[1].set_xlim(-0.05, 1.05)
    axes[1].set_ylim(0, 0.7)
    axes[1].set_xticks(FRACTIONS)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    filepath = FIGURES_DIR / "inoculation_sweep.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def create_individual_plots():
    """Create separate plots for each sweep."""
    for sweep_name, config in SWEEP_CONFIGS.items():
        results = load_sweep_metrics(config["prefix"])
        if not results["fractions"]:
            continue

        fracs = np.array(results["fractions"])

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot both metrics on same axes
        ax.errorbar(
            fracs,
            results["reward_hack"],
            yerr=results["reward_hack_err"],
            label="Reward Hacking Rate",
            color="coral",
            marker="o",
            markersize=8,
            capsize=4,
            linewidth=2,
        )

        ax.errorbar(
            fracs,
            results["accuracy"],
            yerr=results["accuracy_err"],
            label="Accuracy (All Tests)",
            color="steelblue",
            marker="s",
            markersize=8,
            capsize=4,
            linewidth=2,
        )

        ax.errorbar(
            fracs,
            results["first_test"],
            yerr=results["first_test_err"],
            label="First Test Pass Rate",
            color="green",
            marker="^",
            markersize=8,
            capsize=4,
            linewidth=2,
            linestyle="--",
        )

        ax.set_xlabel("Inoculation Fraction", fontsize=12)
        ax.set_ylabel("Rate", fontsize=12)
        ax.set_title(f"{config['label']}: Metrics vs Inoculation Fraction", fontsize=14, fontweight="bold")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, 0.85)
        ax.set_xticks(FRACTIONS)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.legend(loc="center right")
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        filepath = FIGURES_DIR / f"{sweep_name}_sweep.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()


def print_summary():
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("INOCULATION SWEEP RESULTS SUMMARY")
    print("=" * 80)

    for sweep_name, config in SWEEP_CONFIGS.items():
        results = load_sweep_metrics(config["prefix"])
        if not results["fractions"]:
            continue

        print(f"\n{config['label']}:")
        print("-" * 60)
        print(f"{'Fraction':>10} {'First Test':>12} {'All Tests':>12} {'Reward Hack':>12}")
        print("-" * 60)

        for i, frac in enumerate(results["fractions"]):
            ft = results["first_test"][i]
            acc = results["accuracy"][i]
            rh = results["reward_hack"][i]
            print(f"{frac:>10.2f} {ft:>11.1%} {acc:>11.1%} {rh:>11.1%}")


def main():
    print_summary()
    create_sweep_plot()
    create_individual_plots()
    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
