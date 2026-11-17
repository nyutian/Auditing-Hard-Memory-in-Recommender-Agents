from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from schema_ablation import load_jsonl, summarize


ROOT = Path(__file__).resolve().parent


def prefer(generated: list[Path], frozen: list[Path]) -> list[Path]:
    present = [path for path in generated if path.exists()]
    if present:
        return present
    present = [path for path in frozen if path.exists()]
    if not present:
        raise FileNotFoundError(f"none of the expected inputs exists: {generated + frozen}")
    return present


def combine_schema(
    row_paths: list[Path], baseline_paths: list[Path], *, model: str,
) -> dict:
    rows = [row for path in row_paths for row in load_jsonl(path)]
    baseline = [row for path in baseline_paths for row in load_jsonl(path)]
    return {"model": model, "summary": summarize(rows, baseline)}


def errorbar(interval: dict) -> tuple[float, float, float]:
    mean = 100 * interval["mean"]
    return mean, mean - 100 * interval["lo"], 100 * interval["hi"] - mean


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)
    ax.tick_params(axis="both", colors="#222222", width=0.6, length=2.5, pad=2)
    ax.grid(axis="y", color="#E6E6E6", linestyle="--", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT.parent.parent / "figures" / "paper_reproduction",
    )
    parser.add_argument(
        "--frozen-dir", type=Path,
        default=ROOT.parent.parent / "results" / "paper_reproduction",
    )
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    frozen = args.frozen_dir
    summary_output = args.summary_output or frozen / "schema_interface_summary.json"
    languages = ("spanish", "french", "russian", "german")
    gpt = combine_schema(
        prefer(
            [Path("outputs/schema-gpt/schema_rows.jsonl")],
            [frozen / "schema_rows_gpt.jsonl"],
        ),
        prefer(
            [Path("outputs/llm/llm_memories.jsonl")],
            [frozen / "llm_memories_gpt.jsonl"],
        ),
        model="GPT-OSS-120B",
    )
    glm = combine_schema(
        prefer(
            [Path(f"outputs/schema-glm-{language}/schema_rows.jsonl") for language in languages],
            [frozen / f"schema_rows_glm_{language}.jsonl" for language in languages],
        ),
        prefer(
            [Path(f"outputs/llm-glm-{language}/llm_memories.jsonl") for language in languages],
            [frozen / f"llm_memories_glm_{language}.jsonl" for language in languages],
        ),
        model="GLM-5.3-Flash",
    )
    if sum(point["summary"]["conditions"]["hard_schema_committed"]["users"] for point in (gpt, glm)) != 160:
        raise ValueError("schema audit requires 80 users for each model")
    primary_path = prefer(
        [Path("outputs/primary-v3/scripted_summary.json")],
        [frozen / "scripted_summary.json"],
    )[0]
    stress_path = prefer(
        [Path("outputs/stress/stress_summary.json")],
        [frozen / "stress_summary.json"],
    )[0]
    primary = json.loads(primary_path.read_text())
    stress = json.loads(stress_path.read_text())
    combined = {"models": {gpt["model"]: gpt["summary"], glm["model"]: glm["summary"]}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n"
    )

    mpl.rcParams.update({
        "font.size": 7.5,
        "font.family": "serif",
        "font.serif": ["Linux Libertine O", "Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
    })

    fig, axes = plt.subplots(
        1, 3, figsize=(7.1, 2.15),
        gridspec_kw={"width_ratios": [1.05, 1.22, 1.0]},
    )

    # Panel (a): Writing Interface (Model comparison palette: Navy & Terracotta)
    ax = axes[0]
    style_axis(ax)
    conditions = [
        ("hard_schema_committed", "Hard-only\n(committed)"),
        ("calibrated_established", "Calibrated\n(established)"),
        ("calibrated_plausible", "Calibrated\n(plausible)"),
    ]
    x = np.arange(len(conditions), dtype=float)
    width = 0.32
    model_specs = [
        (gpt, "GPT-OSS-120B", "#2B4C7E", "#1B3358"),
        (glm, "GLM-5.3-Flash", "#C85A32", "#8F3B1B"),
    ]
    for offset, (model, label, facecolor, edgecolor) in zip((-width / 2, width / 2), model_specs):
        intervals = [
            model["summary"]["strata"]["ambiguous"][condition]["proxy_rate_wilson95"]
            for condition, _ in conditions
        ]
        values, lowers, uppers = zip(*(errorbar(interval) for interval in intervals))
        ax.bar(
            x + offset, values, width=width, color=facecolor, edgecolor=edgecolor,
            linewidth=0.5, label=label, zorder=3,
        )
        ax.errorbar(
            x + offset, values, yerr=np.asarray([lowers, uppers]), fmt="none",
            ecolor="#222222", linewidth=0.75, capsize=1.8, capthick=0.7, zorder=4,
        )
    ax.set_xticks(x, [label for _, label in conditions])
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Proxy rate in ambiguous histories (%)")
    ax.set_title("(a) Writing interface", fontweight="bold", pad=6)
    ax.legend(
        frameon=True, facecolor="white", edgecolor="#DDDDDD", framealpha=0.92,
        fontsize=6.2, loc="upper center", bbox_to_anchor=(0.48, 1.01), ncol=2,
        handlelength=1.1, columnspacing=0.9, borderpad=0.3,
    )

    # Panel (b): Serving after shift (Grouped bars: Task success vs. Proxy retained)
    ax = axes[1]
    style_axis(ax)
    policies = ["hard_filter", "soft_rank", "transient_relax", "relax_delete", "clarify_once"]
    labels = ["Hard\nfilter", "Soft\nrank", "Transient\nrelax", "Update-1", "Ask\nonce"]
    success = [100 * primary["summary"][policy]["shift"]["success"]["mean"] for policy in policies]
    lo = [100 * primary["summary"][policy]["shift"]["success"]["lo"] for policy in policies]
    hi = [100 * primary["summary"][policy]["shift"]["success"]["hi"] for policy in policies]
    retained = [100.0, 100.0, 100.0, 0.0, 0.0]

    xpos = np.arange(len(policies), dtype=float)
    bwidth = 0.34
    ax.bar(
        xpos - bwidth / 2, success, width=bwidth, color="#2A7B6F", edgecolor="#1B544B",
        linewidth=0.5, label="Task success", zorder=3,
    )
    ax.errorbar(
        xpos - bwidth / 2, success,
        yerr=np.asarray([np.asarray(success) - lo, np.asarray(hi) - success]),
        fmt="none", ecolor="#222222", linewidth=0.75, capsize=1.6, capthick=0.7, zorder=4,
    )
    ax.bar(
        xpos + bwidth / 2, retained, width=bwidth, color="#8C7AA9", edgecolor="#594A73",
        linewidth=0.5, hatch="///", label="Proxy retained", zorder=3,
    )
    ax.set_xticks(xpos, labels)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Rate across users / visits (%)")
    ax.set_title("(b) Serving after catalog shift", fontweight="bold", pad=6)
    ax.legend(
        frameon=True, facecolor="white", edgecolor="#DDDDDD", framealpha=0.92,
        fontsize=6.2, loc="upper right", bbox_to_anchor=(1.0, 1.01), ncol=2,
        handlelength=1.2, columnspacing=0.9, borderpad=0.3,
    )

    # Panel (c): Update safety under verdict noise (Risk gradient palette)
    ax = axes[2]
    style_axis(ax)
    line_specs = {
        1: ("#B8332A", "o", "-", "Update-1 (k=1)"),
        2: ("#D9822B", "s", "--", "Update-2 (k=2)"),
        3: ("#1B7864", "^", "-", "Update-3 (k=3)"),
    }
    for confirmations in (1, 2, 3):
        color, marker, linestyle, label = line_specs[confirmations]
        points = [
            row for row in stress["summary"]
            if row["truth"] == "both" and row["branch"] == "shift"
            and row["confirmations"] == confirmations
        ]
        xs = [100 * row["noise"] for row in points]
        ys = [100 * row["false_deletion"] for row in points]
        ax.plot(
            xs, ys, marker=marker, markersize=3.8, linestyle=linestyle,
            linewidth=1.25, color=color, label=label, zorder=4,
        )
        if xs and ys:
            y_offset = 1.2 if confirmations != 3 else 1.5
            ax.annotate(
                f"{ys[-1]:.1f}%", xy=(xs[-1], ys[-1]),
                xytext=(xs[-1] - 0.15, ys[-1] + y_offset),
                fontsize=6.0, color=color, fontweight="bold", ha="right",
            )
    ax.set_xlim(-0.5, 10.8)
    ax.set_ylim(0, 50)
    ax.set_xticks([0, 5, 10])
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_xlabel("Verdict flip noise (%)")
    ax.set_ylabel("Users with false deletion (%)")
    ax.set_title("(c) Update safety (conjunctive truth)", fontweight="bold", pad=6)
    ax.legend(
        frameon=True, facecolor="white", edgecolor="#DDDDDD", framealpha=0.92,
        fontsize=6.2, loc="upper left", borderpad=0.35, handlelength=1.6,
    )

    fig.tight_layout(pad=0.35, w_pad=0.9)
    fig.savefig(args.output_dir / "interface_audit.pdf", bbox_inches="tight")
    fig.savefig(args.output_dir / "interface_audit.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(summary_output)
    print(args.output_dir / "interface_audit.pdf")


if __name__ == "__main__":
    main()
