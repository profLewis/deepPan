#!/usr/bin/env python3
"""
Generate the alignment-plug STL used to lock adjacent outer-piece wedge halves
together at each strut plane.

The plug is a HORIZONTAL peg: its axis is TANGENTIAL to the strut plane.
The user pushes it from the cavity side into a 4.2 mm channel that spans
both mating half-wedges (10 mm in each half = 20 mm total).  The plug
slip-fits (4.0 mm peg into 4.2 mm channel) so friction holds it in place.

12 plugs are needed in total: 2 per strut plane × 6 strut planes.

Output:
    pipeline_output/plug.stl   (print 12 of these)
"""
from pathlib import Path
import trimesh


PLUG_DIAMETER = 4.0
PLUG_LENGTH   = 18.0   # slightly shorter than the 20 mm channel so it fully fits
SEGMENTS      = 48

OUT_DIR = Path("pipeline_output")
OUT_DIR.mkdir(exist_ok=True)


def main():
    plug = trimesh.creation.cylinder(
        radius=PLUG_DIAMETER / 2.0,
        height=PLUG_LENGTH,
        sections=SEGMENTS,
    )
    # Cylinder is along Z by default, which is what we want (vertical plug).

    stl_path = OUT_DIR / "plug.stl"
    obj_path = OUT_DIR / "plug.obj"
    plug.export(stl_path)
    plug.export(obj_path)
    print(f"Plug: Ø{PLUG_DIAMETER} mm × {PLUG_LENGTH} mm")
    print(f"Wrote {stl_path}  ({len(plug.vertices)} verts, {len(plug.faces)} faces)")
    print(f"Wrote {obj_path}")


if __name__ == "__main__":
    main()
