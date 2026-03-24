"""
main.py — CLI entry: single prompt (pretty trace) or CSV batch (see utils/process_csv.py).
"""

from __future__ import annotations

from utils.logging_setup import configure_logging, prepare_hf_console

# Before importing pipeline / transformers (Hub tqdm, httpx-style chatter).
prepare_hf_console()

import argparse
import logging
from pathlib import Path

import pipeline
import stage2_classifier as s2
import stage3_deep_scan as s3
from utils.cli_output import print_result
from utils.process_csv import run_csv_batch

logger = logging.getLogger("main")


def run_single(prompt: str, verbose: bool) -> None:
    pr = pipeline.run(prompt)
    print_result(prompt, pr, verbose=verbose)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prompt Injection Detection Guard — 4-stage cascade pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prompt", "-p", type=str, help="Single prompt to test")
    mode.add_argument(
        "--csv",
        "-c",
        type=Path,
        help="CSV batch (text column from utils.config / 'text' / 'prompts'; optional 'label' 0/1)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="CSV mode: predictions filename (same folder as input CSV) or absolute path",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Single-prompt mode only: extra stage detail",
    )
    parser.add_argument(
        "--preload-all",
        action="store_true",
        help="Load Stage 2 and 3 models at startup",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="App log level (default: INFO). Hub/HTTP libs are quieter (WARNING; ERROR if you pick ERROR).",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(getattr(logging, args.log_level))

    if args.preload_all:
        logger.info("Preloading Stage 2 and Stage 3 models...")
        s2.preload()
        s3.preload()

    if args.prompt:
        run_single(args.prompt, verbose=args.verbose)
    else:
        if args.verbose:
            logger.warning("CSV mode ignores --verbose; use --prompt for a detailed trace.")
        run_csv_batch(args.csv, predictions_path_user=args.output)


if __name__ == "__main__":
    main()
