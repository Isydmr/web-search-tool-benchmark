#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a minimal GitHub Pages site bundle.")
    parser.add_argument("--output-dir", default="site", help="Output directory for the built Pages site")
    return parser.parse_args()


def copy_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    shutil.copytree(source, destination, dirs_exist_ok=True)


def main() -> None:
    args = parse_args()
    output_dir = (ROOT_DIR / args.output_dir).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(ROOT_DIR / "index.html", output_dir / "index.html")

    nojekyll = ROOT_DIR / ".nojekyll"
    if nojekyll.exists():
        shutil.copy2(nojekyll, output_dir / ".nojekyll")
    else:
        (output_dir / ".nojekyll").write_text("\n", encoding="utf-8")

    copy_tree(ROOT_DIR / "viewer", output_dir / "viewer")

    published_data_dir = ROOT_DIR / "data" / "latest"
    target_data_dir = output_dir / "data" / "latest"
    target_data_dir.mkdir(parents=True, exist_ok=True)
    copy_tree(published_data_dir, target_data_dir)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "viewer": str(output_dir / "viewer" / "index.html"),
                "dataset": str(target_data_dir),
            }
        )
    )


if __name__ == "__main__":
    main()
