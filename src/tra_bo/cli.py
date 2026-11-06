"""Command-line interface for TRA-BO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .llm import FixedAcquisitionSelector, LLMAcquisitionSelector
from .optimizer import TRABO, TRABOConfig


def _load_matrix(path: str) -> NDArray[np.float64]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        values = np.load(source, allow_pickle=False)
    elif suffix == ".json":
        values = np.asarray(json.loads(source.read_text(encoding="utf-8")))
    elif suffix in {".csv", ".txt"}:
        delimiter = "," if suffix == ".csv" else None
        values = np.loadtxt(source, delimiter=delimiter)
    else:
        raise ValueError("matrix files must use .npy, .json, .csv, or .txt")
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError(f"{source} must contain a finite two-dimensional matrix")
    return matrix


def _load_indices(path: str | None) -> list[int] | None:
    if path is None:
        return None
    source = Path(path)
    if source.suffix.lower() == ".json":
        values = json.loads(source.read_text(encoding="utf-8"))
    else:
        values = np.loadtxt(source, dtype=int).reshape(-1).tolist()
    return [int(value) for value in values]


def _selector(name: str):
    return LLMAcquisitionSelector() if name == "llm" else FixedAcquisitionSelector(name)


def _run_optimize(args: argparse.Namespace) -> int:
    candidates = _load_matrix(args.candidates)
    validation_losses = _load_matrix(args.validation_losses)
    config = TRABOConfig(
        budget=args.budget,
        n_initial=args.n_initial,
        recent_count=args.recent_count,
        block_size=args.block_size,
        eta=args.eta,
        xi=args.xi,
        beta=args.beta,
        gp_noise=args.gp_noise,
        gp_restarts=args.gp_restarts,
        random_state=args.seed,
    )
    result = TRABO(config=config, selector=_selector(args.selector)).optimize_precomputed(
        candidates,
        validation_losses,
        initial_indices=_load_indices(args.initial_indices),
    )
    result.save_json(args.output)
    print(
        json.dumps(
            {
                "best_index": result.best_index,
                "best_candidate": result.best_candidate,
                "best_target": result.best_target,
                "final_tra": result.final_tra,
                "evaluations": len(result.observed_indices),
                "output": str(Path(args.output)),
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bayesian optimization with Temporal Rank Agreement"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    optimize = subparsers.add_parser(
        "optimize", help="optimize a finite pool from visible-window losses"
    )
    optimize.add_argument("--candidates", required=True, help="numeric candidate matrix")
    optimize.add_argument(
        "--validation-losses",
        required=True,
        help="aligned matrix containing one column per visible validation window",
    )
    optimize.add_argument("--output", required=True, help="result JSON path")
    optimize.add_argument("--initial-indices", help="optional JSON or text index vector")
    optimize.add_argument(
        "--selector", choices=("llm", "PI", "EI", "UCB", "TS"), default="llm"
    )
    optimize.add_argument("--budget", type=int, default=50)
    optimize.add_argument("--n-initial", type=int, default=16)
    optimize.add_argument("--recent-count", type=int, default=2)
    optimize.add_argument("--block-size", type=int, default=5)
    optimize.add_argument("--eta", type=float, default=1.0 / 8.0)
    optimize.add_argument("--xi", type=float, default=0.01)
    optimize.add_argument("--beta", type=float, default=2.0)
    optimize.add_argument("--gp-noise", type=float, default=1e-6)
    optimize.add_argument("--gp-restarts", type=int, default=2)
    optimize.add_argument("--seed", type=int, default=0)
    optimize.set_defaults(handler=_run_optimize)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
