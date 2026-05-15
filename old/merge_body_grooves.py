#!/usr/bin/env python3
"""
Concatenate pan_holes.obj + grooves into a single open-surface OBJ.

Output: data/quarters/pan_holes_with_grooves.obj
Run thicken_blender.py on this to get a solid that bakes the grooves into
the shell (so there is no seam between pad-hole rim and groove inner edge).
"""
from pathlib import Path

SOURCES = [
    Path("data/quarters/pan_holes.obj"),
    Path("data/grooves/grooves_outer.obj"),
    Path("data/grooves/grooves_central.obj"),
    Path("data/grooves/grooves_inner.obj"),
]
OUT = Path("data/quarters/pan_holes_with_grooves.obj")


def stream_obj(path, v_offset):
    """Yield rewritten OBJ lines from `path` with face indices shifted by v_offset."""
    n_verts = 0
    for line in open(path):
        s = line.rstrip("\n")
        if not s or s.startswith("#"):
            continue
        tok = s.split()
        if tok[0] == "v":
            n_verts += 1
            yield s + "\n"
        elif tok[0] == "f":
            parts = []
            for t in tok[1:]:
                # Keep only positional index (strip vt/vn refs)
                idx = int(t.split("/")[0])
                parts.append(str(idx + v_offset if idx > 0 else idx))
            yield "f " + " ".join(parts) + "\n"
        elif tok[0] in ("o", "g"):
            yield s + "\n"
        # drop vt, vn, mtllib, usemtl
    # Return marker so caller can update the running offset
    return n_verts


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    v_offset = 0
    with open(OUT, "w") as out:
        out.write("# Merged surface: pan_holes + grooves (for solidify)\n")
        # Single `o` block so Blender imports it as one object — otherwise the
        # solidify script only processes the last imported mesh.
        out.write("o pan_holes_with_grooves\n\n")
        for src in SOURCES:
            if not src.exists():
                print(f"WARN: missing {src}, skipping")
                continue
            # Count this file's vertices and emit
            n_verts = 0
            for line in open(src):
                s = line.rstrip("\n")
                if not s or s.startswith("#"):
                    continue
                tok = s.split()
                if tok[0] == "v":
                    n_verts += 1
                    out.write(s + "\n")
                elif tok[0] == "f":
                    parts = []
                    for t in tok[1:]:
                        idx = int(t.split("/")[0])
                        parts.append(str(idx + v_offset))
                    out.write("f " + " ".join(parts) + "\n")
                elif tok[0] in ("o", "g"):
                    # Skip nested group names (we wrote our own `o` above)
                    continue
            v_offset += n_verts
            print(f"  {src.name}: +{n_verts} verts (running total {v_offset})")
    print(f"Wrote {OUT}  ({v_offset} verts)")


if __name__ == "__main__":
    main()
