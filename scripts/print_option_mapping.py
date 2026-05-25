#!/usr/bin/env python3
"""Print seed-dependent letter→text→score mappings (sanity check).

No Hydra, W&B, or model loading. Uses the same shuffle as
``letter_to_score_from_seed`` in ``src/utils/ipi_surrogate.py``.

Examples:
    python scripts/print_option_mapping.py 42 150
    python scripts/print_option_mapping.py --canonical 42
    python scripts/print_option_mapping.py --language en 0 1 2
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.ipi_surrogate import (  # noqa: E402
    IPI_OPTION_LETTERS,
    IPI_OPTION_SCORES,
    IPI_OPTION_TEXT,
    SCORES_ORDERED,
)


def letter_to_score_for_seed(seed: int) -> dict[str, int]:
    """Same assignment as ``letter_to_score_from_seed`` (no logging)."""
    letters = list(IPI_OPTION_LETTERS)
    rng = random.Random(int(seed))
    rng.shuffle(letters)
    return {letter: int(score) for letter, score in zip(letters, SCORES_ORDERED)}


def format_mapping_block(
    letter_to_score: dict[str, int],
    language: str = "pt",
    text_width: int = 30,
) -> str:
    texts = IPI_OPTION_TEXT[language]
    lines = []
    for letter in IPI_OPTION_LETTERS:
        score = letter_to_score[letter]
        text = texts[score]
        lines.append(f"{letter}. {text:<{text_width}} → {score:+d}")
    return "\n".join(lines)


def parse_seeds(raw: list[str]) -> list[int]:
    seeds: list[int] = []
    for item in raw:
        for part in item.replace(",", " ").split():
            part = part.strip()
            if not part:
                continue
            seeds.append(int(part))
    if not seeds:
        raise argparse.ArgumentTypeError("provide at least one seed")
    return seeds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print IPI letter→text→score layout for one or more seeds.",
    )
    parser.add_argument(
        "seeds",
        nargs="+",
        type=str,
        help="Seeds (space- or comma-separated), e.g. 42 150 or 42,150",
    )
    parser.add_argument(
        "--language",
        "-l",
        choices=("pt", "en"),
        default="pt",
        help="Likert option language (default: pt)",
    )
    parser.add_argument(
        "--canonical",
        action="store_true",
        help="Also print the canonical mapping (A=-2 … E=+2, no shuffle)",
    )
    args = parser.parse_args(argv)

    if args.language not in IPI_OPTION_TEXT:
        parser.error(f"unsupported language {args.language!r}")

    seed_list = parse_seeds(args.seeds)

    if args.canonical:
        print("canonical (seed-independent, A→-2 … E→+2)")
        print("-" * 44)
        print(format_mapping_block(dict(IPI_OPTION_SCORES), args.language))
        if seed_list:
            print()

    for i, seed in enumerate(seed_list):
        if i > 0:
            print()
        print(f"seed={seed}")
        print("-" * 44)
        mapping = letter_to_score_for_seed(seed)
        print(format_mapping_block(mapping, args.language))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
