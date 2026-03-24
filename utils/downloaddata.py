"""
Export the SafeGuard prompt-injection dataset from Hugging Face to data/.

Requires: pip install datasets
Run from repo root: python -m utils.downloaddata
"""

from __future__ import annotations

from pathlib import Path

from datasets import load_dataset

_REPO_ROOT = Path(__file__).resolve().parent.parent

dataset_name = "deepset/prompt-injections"
part = "test"

clean_name = dataset_name.replace("/", "_") + "_" + part


_DEFAULT_OUT = _REPO_ROOT / "assets" / "data" / f"{clean_name}.csv"


def export_test_split(out_path: Path | None = None) -> Path:
    path = out_path or _DEFAULT_OUT
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(dataset_name)
    ds[part].to_csv(str(path))
    return path


def main() -> None:
    out = export_test_split()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
