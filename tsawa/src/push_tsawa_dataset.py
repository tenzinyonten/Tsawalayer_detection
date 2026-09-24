#!/usr/bin/env python3
"""Push the local tsawa DatasetDict to the Hugging Face Hub.

Expects an existing ``tsawa/data/processed/tsawa_dataset/`` from
``build_tsawa_dataset.py``. Does not re-tokenize.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_from_disk
from huggingface_hub import HfApi


DEFAULT_REPO = "Yontenn/formatting-tsawa-v1"
DEFAULT_LOCAL = Path("tsawa/data/processed/tsawa_dataset")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload tsawa train/validation/test to the Hub.")
    p.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL)
    p.add_argument("--repo-id", default=DEFAULT_REPO)
    p.add_argument("--private", action="store_true", help="Create/update a private dataset repo.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    local = args.local_dir.expanduser().resolve()
    if not (local / "dataset_dict.json").is_file():
        raise SystemExit(f"No DatasetDict at {local}")

    dset = load_from_disk(str(local))
    expected = {"train": 5463, "validation": 615, "test": 650}
    for split, n in expected.items():
        if split not in dset:
            raise SystemExit(f"missing split {split}")
        if len(dset[split]) != n:
            raise SystemExit(f"{split} has {len(dset[split])} rows, expected {n}")

    print(f"Pushing {args.repo_id}  private={args.private}")
    dset.push_to_hub(args.repo_id, private=args.private)

    api = HfApi()
    readme = local / "HUB_README.md"
    if readme.is_file():
        api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
    label_map = local / "label_map.json"
    if label_map.is_file():
        api.upload_file(
            path_or_fileobj=str(label_map),
            path_in_repo="label_map.json",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
    print(f"https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
