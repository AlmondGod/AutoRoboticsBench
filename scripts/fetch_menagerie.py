from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


REPO_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="aloha", choices=["aloha"])
    parser.add_argument("--dest", default="third_party/mujoco_menagerie")
    args = parser.parse_args()
    fetch_model(model=args.model, dest=Path(args.dest))


def fetch_model(model: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    model_dir = dest / model
    if model_dir.exists():
        print(model_dir)
        return

    tmp = dest / "._clone_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            REPO_URL,
            str(tmp),
        ],
        check=True,
    )
    subprocess.run(["git", "sparse-checkout", "set", model], cwd=tmp, check=True)
    shutil.move(str(tmp / model), str(model_dir))
    shutil.rmtree(tmp)
    print(model_dir)


if __name__ == "__main__":
    main()
