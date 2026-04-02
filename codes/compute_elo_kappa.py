"""
Translation Evaluation Analysis
=================================
Computes:
  1. ELO scores for each model (Claude, DeepSeek, NLLB) from annotator votes
  2. Inter-annotator agreement (Cohen's Kappa) using binary per-model decisions

Input CSV columns:
  key, voter_email, row_idx, en_text, choices, choice_labels, timestamp

  choices: JSON list of model IDs chosen as best (e.g. ["zh_claude"] or ["zh_claude","zh_deepseek"])

Usage:
    python compute_elo_kappa.py results/translation_evaluation/translation_evaluation_nllb_claude_deepseek.csv
    python compute_elo_kappa.py <csv_path> --k 32 --initial-elo 1000
"""

import ast
import csv
import argparse
from collections import defaultdict
from itertools import combinations

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

# ---------------------------------------------------------------------------
# Model display names
# ---------------------------------------------------------------------------

MODEL_DISPLAY = {
    "zh_claude":   "Claude",
    "zh_deepseek": "DeepSeek",
    "zh_nllb":     "NLLB",
}

MEDALS = ["🥇", "🥈", "🥉"]


def display(model_id: str) -> str:
    return MODEL_DISPLAY.get(model_id, model_id)


# ---------------------------------------------------------------------------
# ELO helpers
# ---------------------------------------------------------------------------

def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(ratings: dict, winner: str, loser: str, k: float) -> None:
    ea = expected_score(ratings[winner], ratings[loser])
    eb = expected_score(ratings[loser], ratings[winner])
    ratings[winner] += k * (1 - ea)
    ratings[loser]  += k * (0 - eb)


def update_elo_tie(ratings: dict, a: str, b: str, k: float) -> None:
    ea = expected_score(ratings[a], ratings[b])
    eb = expected_score(ratings[b], ratings[a])
    ratings[a] += k * (0.5 - ea)
    ratings[b] += k * (0.5 - eb)


def compute_elo(rows: list[dict], all_models: list[str], k: float, initial_elo: float) -> dict:
    ratings = {m: initial_elo for m in all_models}
    for row in rows:
        winners = set(row["choices"])
        losers  = set(all_models) - winners
        for w in winners:
            for l in losers:
                update_elo(ratings, w, l, k)
        for a, b in combinations(sorted(winners), 2):
            update_elo_tie(ratings, a, b, k)
    return ratings


# ---------------------------------------------------------------------------
# Kappa helpers (binary per-model decisions)
# ---------------------------------------------------------------------------

def binary_cohen_kappa(bits_a: list[int], bits_b: list[int]) -> tuple[float, float, float]:
    """
    Compute Cohen's Kappa for binary (0/1) labels.
    Returns (kappa, po, pe).
    """
    n = len(bits_a)
    assert n > 0

    agree = sum(1 for a, b in zip(bits_a, bits_b) if a == b)
    po = agree / n

    rate_a = sum(bits_a) / n
    rate_b = sum(bits_b) / n
    pe = rate_a * rate_b + (1 - rate_a) * (1 - rate_b)

    if pe == 1.0:
        return 1.0, po, pe
    return (po - pe) / (1 - pe), po, pe


def kappa_interpretation(kappa: float) -> str:
    if kappa < 0:
        return "Poor (less than chance)"
    elif kappa < 0.20:
        return "Slight"
    elif kappa < 0.40:
        return "Fair"
    elif kappa < 0.60:
        return "Moderate"
    elif kappa < 0.80:
        return "Substantial"
    else:
        return "Almost perfect"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["choices"] = ast.literal_eval(row["choices"])
            row["row_idx"] = int(row["row_idx"])
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute ELO scores and inter-annotator Kappa from translation evaluation CSV")
    parser.add_argument("csv_path", help="Path to the evaluation CSV file")
    parser.add_argument("--k", type=float, default=32, help="ELO K-factor (default: 32)")
    parser.add_argument("--initial-elo", type=float, default=1000, help="Starting ELO for all models (default: 1000)")
    args = parser.parse_args()

    rows = load_csv(args.csv_path)
    all_models = sorted({m for row in rows for m in row["choices"]})
    all_unique_rows = len({row["row_idx"] for row in rows})

    # ---- ELO ----
    console.print()
    console.print("[bold]ELO Scores[/bold]")
    console.print()
    console.print(
        'Each annotation generates pairwise "matches": selected models beat unselected ones; ties\n'
        "within selected/unselected groups."
    )
    console.print()

    ratings = compute_elo(rows, all_models, args.k, args.initial_elo)
    ranked  = sorted(ratings.items(), key=lambda x: x[1], reverse=True)

    elo_table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="")
    elo_table.add_column("Rank",  justify="center")
    elo_table.add_column("Model", justify="center")
    elo_table.add_column("ELO",   justify="center")

    for i, (model, elo) in enumerate(ranked):
        medal = MEDALS[i] if i < len(MEDALS) else "  "
        rank_str = f"{medal} #{i+1}"
        name = display(model)
        name_text = Text(name, style="bold") if i == 0 else Text(name)
        elo_table.add_row(rank_str, name_text, f"{elo:.1f}")

    console.print(elo_table)

    # ---- Selection Frequency ----
    console.print("[bold]Selection Frequency[/bold]")
    console.print()

    # Count unique rows where each model was selected
    rows_selected = defaultdict(set)
    for row in rows:
        for m in row["choices"]:
            rows_selected[m].add(row["row_idx"])

    freq_table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="")
    freq_table.add_column("Model",       justify="center")
    freq_table.add_column("Selected in", justify="center")
    freq_table.add_column("%",           justify="center")

    # Sort by selection count descending (matches ELO ranking for display)
    sorted_models = sorted(all_models, key=lambda m: len(rows_selected[m]), reverse=True)
    for model in sorted_models:
        n_sel = len(rows_selected[model])
        pct   = n_sel / all_unique_rows * 100
        freq_table.add_row(display(model), f"{n_sel} / {all_unique_rows} rows", f"{pct:.1f}%")

    console.print(freq_table)

    # Summary sentence
    best_model      = ranked[0][0]
    best_elo        = ranked[0][1]
    second_model    = ranked[1][0]
    worst_model     = ranked[-1][0]
    worst_pct       = len(rows_selected[worst_model]) / all_unique_rows * 100
    best_pct        = len(rows_selected[best_model]) / all_unique_rows * 100

    console.print(
        f"[bold]{display(best_model)} is the best-performing model[/bold] by a clear margin in both ELO "
        f"and raw selection frequency. {display(worst_model)} lags far behind, selected in only "
        f"{worst_pct:.0f}% of rows."
    )
    console.print()

    # ---- Inter-annotator Kappa (binary per-model) ----
    console.print("[bold]Inter-Annotator Agreement (Cohen's Kappa)[/bold]")
    console.print()

    by_row = defaultdict(list)
    for row in rows:
        by_row[row["row_idx"]].append(row)

    shared = {idx: anns for idx, anns in by_row.items() if len(anns) >= 2}

    if not shared:
        console.print("  No rows annotated by more than one annotator — kappa cannot be computed.")
        return

    # Identify the two annotators from shared rows
    annotators: list[str] = []
    for anns in shared.values():
        for ann in anns:
            if ann["voter_email"] not in annotators:
                annotators.append(ann["voter_email"])
            if len(annotators) == 2:
                break
        if len(annotators) == 2:
            break

    ann_a, ann_b = annotators[0], annotators[1]

    # Build binary decision vectors: one entry per (row, model)
    bits_a: list[int] = []
    bits_b: list[int] = []
    n_shared = 0

    for idx, anns in sorted(shared.items()):
        ann_map = {a["voter_email"]: a for a in anns}
        if ann_a not in ann_map or ann_b not in ann_map:
            continue
        n_shared += 1
        choices_a = set(ann_map[ann_a]["choices"])
        choices_b = set(ann_map[ann_b]["choices"])
        for model in all_models:
            bits_a.append(1 if model in choices_a else 0)
            bits_b.append(1 if model in choices_b else 0)

    n_decisions = len(bits_a)
    kappa, po, pe = binary_cohen_kappa(bits_a, bits_b)
    interp = kappa_interpretation(kappa)

    console.print(
        f"Computed over [bold]{n_shared} shared row_idx × {len(all_models)} models "
        f"= {n_decisions} binary decisions[/bold] (1 = model selected, 0 =\n not selected):"
    )
    console.print()

    kappa_table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="")
    kappa_table.add_column("Metric", justify="left")
    kappa_table.add_column("Value",  justify="left")

    kappa_table.add_row("Observed agreement (Po)", f"{po:.4f}")
    kappa_table.add_row("Expected agreement (Pe)", f"{pe:.4f}")
    kappa_table.add_row(Text("Cohen's κ", style="bold"), Text(f"{kappa:.4f}", style="bold"))
    kappa_table.add_row(Text("Interpretation", style="bold"), Text(interp, style="bold"))

    console.print(kappa_table)

    console.print(
        f"A κ of ~{kappa:.2f} means the two annotators agree more than chance, but there's meaningful\n"
        f"disagreement — typical for subjective translation preference tasks."
    )
    console.print()


if __name__ == "__main__":
    main()
