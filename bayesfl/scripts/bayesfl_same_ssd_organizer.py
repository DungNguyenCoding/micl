#!/usr/bin/env python3
"""Reversible, same-filesystem organization of approved historical BayesFL outputs.

Only exact archive-candidate entries from the verified October 26 dry-run CSV are
moved. This is NOT a backup and DOES NOT reclaim disk space.

Commands: --preview, --move, --verify, --restore
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime

PLAN_FILENAME = "bayesfl_cleanup_DRY_RUN_20261026_194253.csv"
PLAN_SHA256 = "e7d2f3c3d93b87a8649fa6f166789d62fac458d7fe0459ac1f2191545242fb0c"
EXPECTED_ARCHIVE = 121
EXPECTED_PROTECTED = 42


def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def scan_directory(folder):
    """Return logical file bytes and regular-file count; reject symlinks."""
    if folder.is_symlink() or not folder.is_dir():
        raise RuntimeError(f"Not an ordinary directory: {folder}")
    total = count = 0
    for parent, dirs, files in os.walk(folder, followlinks=False):
        for name in dirs:
            if (Path(parent) / name).is_symlink():
                raise RuntimeError(f"Symlink in experiment: {Path(parent)/name}")
        for name in files:
            p = Path(parent) / name
            if p.is_symlink() or not p.is_file():
                raise RuntimeError(f"Unexpected file type: {p}")
            st = p.stat()
            total += st.st_size
            count += 1
    return total, count


def metric_round(directory):
    path = directory / "metrics/global_metrics.csv"
    if not path.is_file():
        return "-"
    final = "-"
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("round"):
                final = str(int(float(row["round"])))
    return final


def checkpoint_round(directory):
    pointer = directory / "resume/latest.json"
    if not pointer.is_file():
        return "-"
    info = json.loads(pointer.read_text(encoding="utf-8"))
    state = json.loads((pointer.parent / info["state_file"]).read_text(encoding="utf-8"))
    return str(int(state["round_id"]))


def validate_directory(folder, row):
    expected_bytes = int(row["current_bytes"])
    expected_files = int(row["current_files"])
    found_bytes, found_files = scan_directory(folder)
    if (found_bytes, found_files) != (expected_bytes, expected_files):
        raise RuntimeError(
            f"Folder changed: {folder}\n"
            f"  expected bytes/files: {expected_bytes}/{expected_files}\n"
            f"  observed bytes/files: {found_bytes}/{found_files}"
        )
    for field, getter in [
        ("current_metric_round", metric_round),
        ("current_checkpoint_round", checkpoint_round),
    ]:
        expected_round = row[field]
        if expected_round != "-":
            observed_round = getter(folder)
            if observed_round != expected_round:
                raise RuntimeError(
                    f"Progress changed: {folder}\n"
                    f"  {field}: expected {expected_round}, found {observed_round}"
                )
    return found_bytes


def load_plan(plan, strict=True):
    if not plan.is_file():
        raise RuntimeError(f"Missing cleanup CSV: {plan}")
    if strict and file_sha256(plan) != PLAN_SHA256:
        raise RuntimeError("Cleanup CSV has changed; do not use this script with another plan.")
    with plan.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("Empty cleanup CSV")
    paths = []
    for row in rows:
        rel = Path(row["relative_path"])
        if (rel.is_absolute() or len(rel.parts) != 2 or rel.parts[0] != "outputs"
                or any(part in (".", "..") for part in rel.parts)):
            raise RuntimeError(f"Unsafe path in manifest: {rel}")
        paths.append(rel.as_posix())
        if row["status"] != "MATCH" or row["warning"]:
            raise RuntimeError(f"Unsafe status in manifest: {rel}")
        if row["action"] not in ("ARCHIVE_CANDIDATE", "KEEP_PROTECTED"):
            raise RuntimeError(f"Unknown action: {row['action']}")
        if row["current_bytes"] != row["snapshot_bytes"] or row["current_files"] != row["snapshot_files"]:
            raise RuntimeError(f"Manifest records differ: {rel}")
    if len(paths) != len(set(paths)):
        raise RuntimeError("Duplicate manifest paths")
    a = [r for r in rows if r["action"] == "ARCHIVE_CANDIDATE"]
    k = [r for r in rows if r["action"] == "KEEP_PROTECTED"]
    if strict and (len(a), len(k)) != (EXPECTED_ARCHIVE, EXPECTED_PROTECTED):
        raise RuntimeError(f"Unexpected manifest counts: {len(a)} archive, {len(k)} protected")
    return a, k, set(paths)


def destination(archive, row):
    # Archive folders grouped by historical experiment family.
    return archive / row["category"] / Path(row["relative_path"]).name


def preflight(root, archive, archive_rows, keep_rows, known_paths, action):
    if root.resolve() == archive.resolve() or root.resolve() in archive.resolve().parents:
        raise RuntimeError("Archive must be outside the active repository")
    if root.stat().st_dev != (archive if archive.exists() else archive.parent).stat().st_dev:
        raise RuntimeError("Archive is not on the same filesystem; refusing non-atomic move")

    # All protected experiments must remain in place and unchanged.
    for row in keep_rows:
        validate_directory(root / row["relative_path"], row)

    unknown = sorted(
        p.relative_to(root).as_posix()
        for p in (root / "outputs").iterdir()
        if p.is_dir() and ("outputs/" + p.name) not in known_paths
    )
    if unknown and action == "move":
        raise RuntimeError("Unclassified outputs found; review before moving:\n" + "\n".join(unknown))

    needs_move = already_moved = 0
    total = 0
    for row in archive_rows:
        src = root / row["relative_path"]
        dst = destination(archive, row)
        source_present, archived_present = src.exists(), dst.exists()
        if source_present == archived_present:
            raise RuntimeError(
                f"Invalid placement: src={'present' if source_present else 'missing'}, "
                f"archive={'present' if archived_present else 'missing'}\n{src}\n{dst}"
            )
        validate_directory(src if source_present else dst, row)
        total += int(row["current_bytes"])
        needs_move += int(source_present)
        already_moved += int(archived_present)
    return needs_move, already_moved, total, unknown


def run(action, root, plan, archive, strict=True):
    root = root.resolve()
    archive = archive.absolute()
    archive_rows, keep_rows, known_paths = load_plan(plan, strict=strict)
    if not root.is_dir() or not (root / "outputs").is_dir():
        raise RuntimeError(f"Not a BayesFL project directory: {root}")

    pending, archived, total, unknown = preflight(root, archive, archive_rows, keep_rows, known_paths, action)
    print("\n" + "="*78)
    print("BAYESFL SAME-SSD ORGANIZER — " + action.upper())
    print("="*78)
    print("Original :", root)
    print("Archive  :", archive)
    print("Manifest :", plan)
    print("Protected folders verified:", len(keep_rows))
    print("Historical folders total  :", len(archive_rows))
    print("Still in repository       :", pending)
    print("Already in archive        :", archived)
    print(f"Archived logical size     : {total/1024**3:.3f} GiB")
    print("Unknown untouched folders :", len(unknown))
    print("Free disk space gained    : 0 (same SSD)")

    if action == "preview":
        print("\nEXACT PATHS TO MOVE:")
        for row in archive_rows:
            src = root / row["relative_path"]
            if src.exists():
                print(f"  {int(row['current_bytes'])/1024**2:8.1f} MiB  {row['relative_path']}")
        print("\nPASS: PREVIEW ONLY — no files moved")
        return

    if action == "verify":
        if pending != 0:
            raise RuntimeError(f"Not all archived yet: {pending} paths still in repository")
        print("\nPASS: All archive folders validated; protected originals retained")
        return

    if action == "move":
        if pending == 0:
            print("\nAll selected folders are already organized. Nothing to move.")
            return
        archive.mkdir(parents=True, exist_ok=True)
        saved_plan = archive / PLAN_FILENAME
        if saved_plan.exists() and file_sha256(saved_plan) != file_sha256(plan):
            raise RuntimeError("Archive manifest differs; refusing to overwrite")
        if not saved_plan.exists():
            shutil.copy2(plan, saved_plan)
        readme = archive / "README_NOT_A_BACKUP.txt"
        if not readme.exists():
            readme.write_text(
                "This directory was produced by a same-filesystem MOVE from bayesfl/outputs.\n"
                "It is NOT an independent backup and does NOT free space on this SSD.\n"
                "To restore folders, use the organizer's --restore command.\n",
                encoding="utf-8",
            )
        journal = archive / "move_journal.tsv"
        with journal.open("a", encoding="utf-8") as log:
            for row in archive_rows:
                src = root / row["relative_path"]
                dst = destination(archive, row)
                if not src.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    raise RuntimeError(f"Refusing to overwrite archive target: {dst}")
                inode = src.stat().st_ino
                os.rename(src, dst)  # atomic rename, same filesystem
                if src.exists() or dst.stat().st_ino != inode:
                    raise RuntimeError(f"Rename integrity check failed: {src} -> {dst}")
                log.write(f"{datetime.now().isoformat(timespec='seconds')}\tMOVE\t{row['relative_path']}\t{dst}\n")
                log.flush()
                os.fsync(log.fileno())
                print("MOVED:", row["relative_path"])
        preflight(root, archive, archive_rows, keep_rows, known_paths, "verify")
        print("\nPASS: MOVE + VERIFICATION COMPLETE")
        return

    if action == "restore":
        if archived == 0:
            print("\nNothing to restore; all historical folders already in repository")
            return
        journal = archive / "move_journal.tsv"
        with journal.open("a", encoding="utf-8") as log:
            for row in reversed(archive_rows):
                src = root / row["relative_path"]
                dst = destination(archive, row)
                if src.exists():
                    continue
                if not dst.exists():
                    raise RuntimeError(f"Missing archived folder: {dst}")
                src.parent.mkdir(parents=True, exist_ok=True)
                inode = dst.stat().st_ino
                os.rename(dst, src)
                if dst.exists() or src.stat().st_ino != inode:
                    raise RuntimeError(f"Restore integrity check failed: {dst} -> {src}")
                log.write(f"{datetime.now().isoformat(timespec='seconds')}\tRESTORE\t{row['relative_path']}\t{src}\n")
                log.flush()
                os.fsync(log.fileno())
                print("RESTORED:", row["relative_path"])
        preflight(root, archive, archive_rows, keep_rows, known_paths, "preview")
        print("\nPASS: RESTORE COMPLETE")
        return

    raise RuntimeError(f"Unsupported action: {action}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--preview", action="store_true", help="Read-only preflight and exact folder list")
    actions.add_argument("--move", action="store_true", help="Move exact archived folders into sibling directory")
    actions.add_argument("--verify", action="store_true", help="Read-only verification after moving")
    actions.add_argument("--restore", action="store_true", help="Undo the move and restore original paths")
    parser.add_argument("--confirm", default="", help="Required exact phrase for move/restore")
    args = parser.parse_args()
    action = "preview" if args.preview else "move" if args.move else "verify" if args.verify else "restore"
    if action in ("move", "restore"):
        phrase = "MOVE_121_ARCHIVE_CANDIDATES" if action == "move" else "RESTORE_121_ARCHIVE_CANDIDATES"
        if args.confirm != phrase:
            parser.error(f"{action} requires --confirm {phrase}")
    root = Path(__file__).resolve().parents[1]
    plan = root.parent / PLAN_FILENAME
    archive = root.parent / "bayesfl_historical_archive"
    run(action, root, plan, archive)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
