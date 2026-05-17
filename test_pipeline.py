#!/usr/bin/env python3
"""
Verify that the pipeline regenerates the committed definitive model.

Workflow:
  1. The committed reference outputs live in `pipeline_output/`.
  2. This test moves them aside to `pipeline_output_reference/`, runs the
     full pipeline (`./build_from_scratch.sh` + the split/assembly steps),
     and compares the regenerated files against the reference.
  3. For each file:
       * If the regenerated copy is byte-identical to the reference, the
         regenerated copy is REMOVED (no need to keep a duplicate) — and the
         reference is restored to `pipeline_output/`.
       * If the regenerated copy differs, the reference is left in
         `pipeline_output_reference/` and the differing files are reported
         so you can inspect them.

Run:
    python3 test_pipeline.py            # quick checksum-only check
    python3 test_pipeline.py --full     # full regenerate-and-compare (~90 min)
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

OUT_DIR = Path("pipeline_output")
REF_DIR = Path("pipeline_output_reference")
MANIFEST = OUT_DIR / "MANIFEST.sha256"

# Files we treat as the definitive committed model.  When adding parts to
# the model, append the filename here.
TRACKED = (
    ["pan_printable.stl"]
    + [f"pan_piece_outer_{i}.stl" for i in range(6)]
    + ["pan_piece_central.stl"]
    + ["bottom_plate.stl", "plug.stl"]
    + [f"bottom_plate_sector_{i}.stl" for i in range(6)]
    + [f"3mf/{stem}.3mf" for stem in (
        ["pan_central"]
        + [f"pan_outer_{i}" for i in range(6)]
        + [f"bottom_plate_sector_{i}" for i in range(6)]
        + ["small_parts_plugs"]
    )]
)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest():
    """Compute hashes for all TRACKED outputs and write MANIFEST.sha256."""
    manifest = {}
    for rel in TRACKED:
        p = OUT_DIR / rel
        if p.exists():
            manifest[rel] = sha256_file(p)
        else:
            print(f"  WARN: missing tracked file {rel}")
    MANIFEST.parent.mkdir(exist_ok=True)
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"Wrote {MANIFEST} with {len(manifest)} entries")
    return manifest


def quick_check():
    """Fast: just verify pipeline_output/ hashes still match MANIFEST.sha256."""
    if not MANIFEST.exists():
        print(f"No manifest at {MANIFEST}; building one from current outputs.")
        build_manifest()
        return 0
    manifest = json.load(open(MANIFEST))
    mismatches = []
    for rel, want in manifest.items():
        p = OUT_DIR / rel
        if not p.exists():
            mismatches.append(f"{rel}: MISSING")
            continue
        got = sha256_file(p)
        if got != want:
            mismatches.append(f"{rel}: HASH MISMATCH (have {got[:8]}, manifest {want[:8]})")
    if mismatches:
        print("Mismatches:")
        for m in mismatches:
            print(f"  {m}")
        return 1
    print(f"All {len(manifest)} tracked files match the manifest. ✓")
    return 0


def full_regenerate_and_compare():
    """Move committed outputs aside, regenerate, compare each file."""
    if REF_DIR.exists():
        print(f"{REF_DIR} already exists — aborting (remove it manually first)")
        return 1
    if not OUT_DIR.exists():
        print(f"{OUT_DIR} does not exist — nothing to compare against")
        return 1

    print(f"Backing up {OUT_DIR} → {REF_DIR}")
    shutil.move(str(OUT_DIR), str(REF_DIR))
    OUT_DIR.mkdir(exist_ok=True)

    print("Running ./build_from_scratch.sh (this can take ~90 min)...")
    rc = subprocess.run(["./build_from_scratch.sh"], check=False).returncode
    if rc != 0:
        print(f"Build failed (exit code {rc}); leaving {REF_DIR} in place for inspection.")
        return rc

    print("\nComparing regenerated outputs to reference...")
    mismatches = []
    matches = []
    for rel in TRACKED:
        new = OUT_DIR / rel
        ref = REF_DIR / rel
        if not ref.exists():
            print(f"  {rel}: no reference, skipping")
            continue
        if not new.exists():
            mismatches.append(f"{rel}: not regenerated")
            continue
        if sha256_file(new) == sha256_file(ref):
            matches.append(rel)
        else:
            mismatches.append(rel)

    if mismatches:
        print(f"\n{len(mismatches)} files differ — leaving both in place for inspection:")
        for m in mismatches:
            print(f"  ✗ {m}")
        print(f"Reference is in {REF_DIR}; regenerated is in {OUT_DIR}.")
        return 1

    print(f"\nAll {len(matches)} files match. Removing the regenerated copies and "
          f"restoring the reference.")
    for rel in matches:
        (OUT_DIR / rel).unlink()
    # Restore reference dir as the canonical output
    shutil.rmtree(OUT_DIR)
    shutil.move(str(REF_DIR), str(OUT_DIR))
    print("Pipeline is reproducible. ✓")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true",
                    help="Run the full regeneration + byte-for-byte compare "
                         "(SLOW: ~90 min)")
    ap.add_argument("--rebuild-manifest", action="store_true",
                    help="Rebuild MANIFEST.sha256 from current pipeline_output/")
    args = ap.parse_args()

    if args.rebuild_manifest:
        build_manifest()
        return 0
    if args.full:
        return full_regenerate_and_compare()
    return quick_check()


if __name__ == "__main__":
    sys.exit(main())
