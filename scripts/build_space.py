"""Assemble the Hugging Face Space staging directory from canonical sources.

The Space expects a flat layout: the OpenEnv integration modules at the root,
the core packages beside them, and the Dockerfile at the top level. That layout
used to be committed as hf_space_staging/, which duplicated every package in
the repository. This script rebuilds it on demand instead.

Usage:
    python scripts/build_space.py [--out build/space]

Deploy the result with:
    hf upload <owner>/<space> build/space . --repo-type space
"""

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# (source relative to repo root, destination relative to staging root)
COPIES = [
    ("deploy/space/Dockerfile", "Dockerfile"),
    ("deploy/space/README.md", "README.md"),
    ("deploy/space/pyproject.toml", "pyproject.toml"),
    ("deploy/space/requirements.txt", "requirements.txt"),
    ("deploy/space/uv.lock", "uv.lock"),
    ("openenv.yaml", "openenv.yaml"),
    ("nexus", "nexus"),
    ("agents", "agents"),
    ("nexus_openenv/__init__.py", "__init__.py"),
    ("nexus_openenv/client.py", "client.py"),
    ("nexus_openenv/models.py", "models.py"),
    ("nexus_openenv/server/__init__.py", "server/__init__.py"),
    ("nexus_openenv/server/app.py", "server/app.py"),
    ("nexus_openenv/server/nexus_environment.py", "server/nexus_environment.py"),
]


def build(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for src_rel, dst_rel in COPIES:
        src = REPO_ROOT / src_rel
        dst = out_dir / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(
                src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
            )
        else:
            shutil.copy2(src, dst)
    print(f"Staging directory written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="build/space", help="output directory")
    args = parser.parse_args()
    build((REPO_ROOT / args.out).resolve())
