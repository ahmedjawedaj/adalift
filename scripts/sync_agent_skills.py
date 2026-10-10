from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".agents" / "skills"
DEST = ROOT / ".claude" / "skills"


def files_match(source: Path, destination: Path) -> bool:
    if not destination.exists():
        return False
    source_files = sorted(p.relative_to(source) for p in source.rglob("*") if p.is_file())
    dest_files = sorted(p.relative_to(destination) for p in destination.rglob("*") if p.is_file())
    if not set(source_files).issubset(dest_files):
        return False
    return all(filecmp.cmp(source / rel, destination / rel, shallow=False) for rel in source_files)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mirror canonical Codex skills into Claude project skills."
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        if not files_match(SOURCE, DEST):
            raise SystemExit("Agent skills are out of sync. Run: make sync-skills")
        print("agent skills synchronized")
        return
    for source in SOURCE.iterdir():
        if source.is_dir():
            shutil.copytree(source, DEST / source.name, dirs_exist_ok=True)
    print(f"synced {SOURCE.relative_to(ROOT)} -> {DEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
