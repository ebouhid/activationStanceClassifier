import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes
import json
from math import pi
from pathlib import Path
from scipy import stats

import matplotlib

matplotlib.use("Agg")
# import numpy as np

# --- Configuration ---
FILE_BASELINE = "runs/2026-02-03/10-47-56/metrics_20260203_104851.json"
FILE_INTERVENED = "runs/2026-02-02/14-44-47/metrics_20260202_144539.json"
FILE_BASELINE_PIS = "runs/2026-02-03/10-47-56/pair_results_20260203_104851.csv"
FILE_INTERVENED_PIS = "runs/2026-02-02/14-44-47/pair_results_20260202_144539.csv"


def load_and_identify(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Heuristic: The one with larger absolute PI is the Intervention
    pi = data['model_polarization_index']
    return data, pi


def load_question_pis(baseline_csv, intervened_csv):
    """Load question-level PIs from CSV files."""
    baseline_df = pd.read_csv(baseline_csv)
    intervened_df = pd.read_csv(intervened_csv)
    return baseline_df, intervened_df


def create_boxplot_comparison(baseline_pis, intervened_pis, output_dir):
    """Creates a boxplot comparing question-level PI distributions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    data_to_plot = [baseline_pis, intervened_pis]
    bp = ax.boxplot(data_to_plot, tick_labels=['Baseline', 'Intervened'],
                    patch_artist=True, widths=0.6,
                    orientation='horizontal')

    # Color the boxes
    colors = ['blue', 'red']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Polarization Index (PI)', fontsize=12)
    ax.set_title('Distribution of Question-Level PI', fontsize=14)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)

    # Add statistics as text
    baseline_mean = baseline_pis.mean()
    intervened_mean = intervened_pis.mean()
    baseline_std = baseline_pis.std()
    intervened_std = intervened_pis.std()

    # Perform Wilcoxon Signed-Rank Test
    wilcoxon_stat, wilcoxon_pvalue = stats.wilcoxon(
        baseline_pis, intervened_pis)
    effect_size = (intervened_mean - baseline_mean) / \
        baseline_std if baseline_std != 0 else float('inf')

    stats_text = f"Baseline: μ={baseline_mean:.3f}, σ={baseline_std:.3f}\n"
    stats_text += f"Intervened: μ={intervened_mean:.3f}, σ={intervened_std:.3f}\n"
    stats_text += f"Wilcoxon: p={wilcoxon_pvalue:.4f} (stat={wilcoxon_stat:.3f})\n"
    stats_text += f"Effect Size (Rank-Biserial r): {effect_size:.3f}"

    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    boxplot_path = output_dir / "pi_shift_boxplot.png"
    fig.savefig(boxplot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return boxplot_path, {
        'baseline_mean': baseline_mean,
        'baseline_std': baseline_std,
        'intervened_mean': intervened_mean,
        'intervened_std': intervened_std,
        'wilcoxon_statistic': float(wilcoxon_stat),
        'wilcoxon_pvalue': float(wilcoxon_pvalue)
    }


def create_parallel_coordinates_plot(baseline_pis, intervened_pis, output_dir):
    """Creates a parallel coordinates plot showing shift from baseline to intervened."""
    fig, ax = plt.subplots(figsize=(10, 8))

    n_questions = len(baseline_pis)

    # Determine color based on direction and magnitude of change
    changes = intervened_pis - baseline_pis

    # Normalize colors based on change magnitude
    max_change = max(abs(changes.min()), abs(changes.max()))

    for i in range(n_questions):
        change = changes[i]
        if change < 0:
            # color = plt.cm.Reds(min(abs(change) / max_change, 1.0))
            color = 'red'
        else:
            # color = plt.cm.Blues(min(abs(change) / max_change, 1.0))
            color = 'blue'

        ax.plot([0, 1], [baseline_pis[i], intervened_pis[i]],
                color=color, alpha=0.4, linewidth=0.8)

    # Add vertical axes
    ax.axvline(0, color='black', linewidth=2, label='Baseline')
    ax.axvline(1, color='black', linewidth=2, label='Intervened')

    # Add horizontal zero line
    ax.axhline(0, color='gray', linewidth=1, linestyle='--', alpha=0.5)

    # Customize plot
    ax.set_xlim(-0.1, 1.1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Baseline', 'Intervened'], fontsize=12)
    ax.set_ylabel('Polarization Index (PI)', fontsize=12)
    ax.set_title('Question-Level PI Shift: Baseline → Intervened', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Add summary statistics
    baseline_mean = baseline_pis.mean()
    intervened_mean = intervened_pis.mean()
    mean_shift = intervened_mean - baseline_mean

    # Draw mean lines
    ax.plot([0, 1], [baseline_mean, intervened_mean],
            color='black', linewidth=3, linestyle='-', label='Mean', zorder=10)

    # Add legend with color explanation
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.6, label='Leftward shift'),
        Patch(facecolor='blue', alpha=0.6, label='Rightward shift'),
        plt.Line2D([0], [0], color='black', linewidth=3, label='Mean PI')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    # Add text box with statistics
    stats_text = f"Mean shift: {mean_shift:.3f}\n"
    stats_text += f"Questions: {n_questions}"
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    fig.tight_layout()
    parallel_path = output_dir / "pi_shift_parallel.png"
    fig.savefig(parallel_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return parallel_path


def create_radar_chart(categories, baseline_vals, intervened_vals, title):
    """Generates a Radar Chart comparing Baseline vs Intervened."""
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    # Initialise the spider plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    assert isinstance(ax, PolarAxes)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, color='grey', size=10)

    # Draw ylabels (concentric circles)
    ax.set_rlabel_position(0)
    plt.yticks([-4, -2, 0, 2, 4], ["-4", "-2", "0",
               "2", "4"], color="black", size=10)
    plt.ylim(-4, 4)
    zero_angles = [i / 360 * 2 * pi for i in range(361)]
    ax.plot(zero_angles, [0] * len(zero_angles),
            color="black", linewidth=1.2, alpha=0.7)

    # Plot Baseline
    values_b = baseline_vals + baseline_vals[:1]
    ax.plot(angles, values_b, linewidth=1,
            linestyle='dashed', label='Baseline')
    ax.fill(angles, values_b, 'b', alpha=0.1)

    # Plot Intervened
    values_i = intervened_vals + intervened_vals[:1]
    ax.plot(angles, values_i, linewidth=2, linestyle='solid',
            color='red', label='Intervened')
    ax.fill(angles, values_i, 'r', alpha=0.1)

    plt.title(title, size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    return fig


def main():
    output_dir = Path(FILE_INTERVENED).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    data1, pi1 = load_and_identify(FILE_BASELINE)
    data2, pi2 = load_and_identify(FILE_INTERVENED)

    # Identify which is which
    if abs(pi1) > abs(pi2):
        intervened, baseline = data1, data2
    else:
        intervened, baseline = data2, data1

    print(f"Baseline PI: {baseline['model_polarization_index']:.3f}")
    print(f"Intervened PI: {intervened['model_polarization_index']:.3f}")

    # 2. Prepare Data for Radar Chart (Axis Breakdown)
    axes_data = {}
    for axis, stats in baseline['by_axis'].items():
        axes_data[axis] = {'Baseline': stats['mean_pi']}

    for axis, stats in intervened['by_axis'].items():
        if axis in axes_data:
            axes_data[axis]['Intervened'] = stats['mean_pi']

    df_axes = pd.DataFrame(axes_data).T

    # Sort by 'Intervened' magnitude to make the chart look organized
    df_axes['diff'] = (df_axes['Intervened'] - df_axes['Baseline']).abs()
    df_axes = df_axes.sort_values('diff', ascending=False)

    # 3. Plot 1: Radar Chart (The "Footprint")
    fig1 = create_radar_chart(
        df_axes.index.tolist(),
        df_axes['Baseline'].tolist(),
        df_axes['Intervened'].tolist(),
        "PI Breakdown by Theme"
    )
    fig1_path = output_dir / "pi_shift_radar.png"
    fig1.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # 4. Plot 2: The "Fluidity" Spectrum (Bar Chart of Deltas)
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # Calculate change
    change = df_axes['Intervened'] - df_axes['Baseline']
    colors = ['red' if x < 0 else 'blue' for x in change]  # Red for Left Shift

    change.plot(kind='barh', ax=ax2, color=colors, alpha=0.7)
    ax2.axvline(0, color='black', linewidth=0.8)
    ax2.set_xlabel("Shift in PI (Negative = Move Left)", fontsize=12)
    ax2.set_title("Fluidity/PI Shift by Theme", fontsize=14)
    ax2.grid(axis='x', linestyle='--', alpha=0.5)

    # Add labels
    for i, v in enumerate(change):
        ax2.text(v - 0.1 if v < 0 else v + 0.05, i, f"{v:.2f}", va='center')
    fig2_path = output_dir / "pi_shift_fluidity.png"
    fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # 6. Plot 3: Question-Level PI Boxplot (if CSV files provided)
    boxplot_path = None
    boxplot_stats = None
    parallel_path = None
    if FILE_BASELINE_PIS and FILE_INTERVENED_PIS:
        baseline_df, intervened_df = load_question_pis(
            FILE_BASELINE_PIS, FILE_INTERVENED_PIS)
        baseline_pis = baseline_df['polarization_index'].values
        intervened_pis = intervened_df['polarization_index'].values
        boxplot_path, boxplot_stats = create_boxplot_comparison(
            baseline_pis, intervened_pis, output_dir)
        print(f"\nQuestion-Level PI Statistics:")
        print(
            f"Baseline - Mean: {boxplot_stats['baseline_mean']:.3f}, Std: {boxplot_stats['baseline_std']:.3f}")
        print(
            f"Intervened - Mean: {boxplot_stats['intervened_mean']:.3f}, Std: {boxplot_stats['intervened_std']:.3f}")
        print(f"\nWilcoxon Signed-Rank Test:")
        print(f"Statistic: {boxplot_stats['wilcoxon_statistic']:.4f}")
        print(f"P-value: {boxplot_stats['wilcoxon_pvalue']:.4f}")
        if boxplot_stats['wilcoxon_pvalue'] < 0.05:
            print("Result: Statistically significant difference (p < 0.05)")
        else:
            print("Result: No statistically significant difference (p >= 0.05)")

        # 7. Plot 4: Parallel Coordinates Plot
        parallel_path = create_parallel_coordinates_plot(
            baseline_pis, intervened_pis, output_dir)
        print(f"\nParallel coordinates plot saved to: {parallel_path}")

    df_axes.to_csv(output_dir / "pi_shift_axes.csv")
    summary = {
        "baseline_pi": baseline["model_polarization_index"],
        "intervened_pi": intervened["model_polarization_index"],
        "baseline_std": baseline["pi_std"],
        "intervened_std": intervened["pi_std"],
        "FILE_BASELINE": str(Path(FILE_BASELINE)),
        "FILE_INTERVENED": str(Path(FILE_INTERVENED)),
        "file_baseline_pis": str(Path(FILE_BASELINE_PIS)) if FILE_BASELINE_PIS else None,
        "file_intervened_pis": str(Path(FILE_INTERVENED_PIS)) if FILE_INTERVENED_PIS else None,
        "output_dir": str(output_dir),
        "artifacts": {
            "radar": str(fig1_path),
            "fluidity": str(fig2_path),
            "boxplot": str(boxplot_path),
            "parallel": str(parallel_path) if parallel_path else None,
            "axes_csv": str(output_dir / "pi_shift_axes.csv"),
        },
    }
    if boxplot_path:
        summary["artifacts"]["boxplot"] = str(boxplot_path)
        summary["question_level_stats"] = boxplot_stats
    if parallel_path:
        summary["artifacts"]["parallel"] = str(parallel_path)

    with open(output_dir / "pi_shift_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
