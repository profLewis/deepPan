#!/usr/bin/env python3
"""
Generate a flat circular bottom plate to attach to the drum's skirt.

Plate sits at Y = SKIRT_BOTTOM_Y (drum-skirt bottom rim, ≈ -123 mm) and is
slightly larger than the skirt outer radius. N (default 8) M4 countersunk
through-holes evenly spaced near the rim — each outer split-piece gets
matching tapped/heat-set holes baked in by the splitting step.

Output:
    pipeline_output/bottom_plate.obj
    pipeline_output/bottom_plate.stl
    pipeline_output/bottom_plate_screw_pattern.json   (positions used by the
        split step to drill matching holes in the pan skirts)
"""
from pathlib import Path
import argparse
import json
import math
import numpy as np
import trimesh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skirt-r",      type=float, default=246.0,
                   help="drum skirt outer radius (mm)")
    p.add_argument("--plate-margin", type=float, default=8.0,
                   help="plate radius = skirt_r + this (mm)")
    p.add_argument("--thickness",    type=float, default=6.0,
                   help="plate thickness (mm)")
    p.add_argument("--n-holes",      type=int,   default=8,
                   help="number of evenly-spaced M4 holes")
    p.add_argument("--hole-r-clear", type=float, default=2.2,
                   help="M4 clearance hole radius (mm)")
    p.add_argument("--counter-r",    type=float, default=4.2,
                   help="countersink radius at top face (mm)")
    p.add_argument("--counter-depth",type=float, default=2.5,
                   help="countersink depth (mm)")
    p.add_argument("--hole-circle-margin", type=float, default=10.0,
                   help="hole circle radius = skirt_r - this (mm); "
                        "holes sit inboard of the skirt outer edge")
    p.add_argument("--skirt-y", type=float, default=-122.0,
                   help="Y of drum-skirt bottom rim (mm); plate top sits here. "
                        "Matches the body truncation level in split_assembly.py.")
    p.add_argument("--out-dir", default="pipeline_output")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    plate_r = args.skirt_r + args.plate_margin
    hole_circle_r = args.skirt_r - args.hole_circle_margin
    y_top = args.skirt_y
    y_bot = y_top - args.thickness

    # The exported STL is Z-up (Blender convention): drum's vertical axis = Z.
    # Plate sits at the drum's skirt bottom (Z ≈ −123). z_top = top face of
    # plate (touching skirt); z_bot = bottom face.
    z_top = args.skirt_y
    z_bot = z_top - args.thickness
    plate_center_z = (z_top + z_bot) / 2

    print(f"Plate: outer R={plate_r}mm, thickness={args.thickness}mm")
    print(f"  Z top  = {z_top:.2f} (sits flush against drum skirt bottom)")
    print(f"  Z bot  = {z_bot:.2f}")
    print(f"  {args.n_holes} M4 countersunk holes at R = {hole_circle_r:.2f}mm")

    # trimesh.creation.cylinder is along Z by default — that's already the
    # axis we want. No rotation needed.
    plate = trimesh.creation.cylinder(
        radius=plate_r,
        height=args.thickness,
        sections=128,
    )
    plate.apply_translation([0, 0, plate_center_z])

    # Hole positions in the XY plane
    holes = []
    for i in range(args.n_holes):
        ang = 2 * math.pi * i / args.n_holes
        # Stagger so holes don't sit on strut planes (15+30·i deg).
        ang += math.radians(22.5)
        x = hole_circle_r * math.cos(ang)
        y = hole_circle_r * math.sin(ang)
        holes.append({
            "x": x, "y": y,
            "angle_deg": math.degrees(ang),
            "clear_r": args.hole_r_clear,
            "counter_r": args.counter_r,
            "counter_depth": args.counter_depth,
        })

    for h in holes:
        # Through-hole, vertical along Z
        bore = trimesh.creation.cylinder(
            radius=h["clear_r"], height=args.thickness + 1.0, sections=32,
        )
        bore.apply_translation([h["x"], h["y"], plate_center_z])
        plate = plate.difference(bore)
        # Countersink on top face (the face touching the drum is the top)
        ctr = trimesh.creation.cylinder(
            radius=h["counter_r"],
            height=args.counter_depth + 0.5,
            sections=32,
        )
        ctr.apply_translation([h["x"], h["y"],
                               z_top - args.counter_depth / 2 + 0.25])
        plate = plate.difference(ctr)
        print(f"  hole at ({h['x']:+7.2f}, {h['y']:+7.2f})  "
              f"angle={h['angle_deg']:6.2f}deg")

    # ── Strut joinery: 12 wedge-screw clearance holes ──
    # (Plugs are now HORIZONTAL pegs between adjacent half-wedges, so the
    #  bottom plate no longer needs vertical plug pass-throughs.)
    # Positions MUST match split_assembly.py constants.
    STRUT_ANGLES_DEG = [15.0, 75.0, 135.0, 195.0, 255.0, 315.0]
    TAP_HOLE_R       = 170.0
    TAP_OFFSET       = 5.0    # tangential offset from strut plane (matches split_assembly.py)
    WEDGE_SCREW_CLEAR_R    = 2.25  # 4.5mm Ø clearance for M4
    WEDGE_SCREW_COUNTER_R  = 4.2   # 8.4mm Ø countersink for M4 flat head
    WEDGE_SCREW_COUNTER_DEPTH = 2.5

    print("\n  ── 12 wedge-screw holes (M4 from below, countersunk on underside) ──")

    def cut_through(plate, x, y, r):
        bore = trimesh.creation.cylinder(
            radius=r, height=args.thickness + 2.0, sections=32,
        )
        bore.apply_translation([x, y, plate_center_z])
        return plate.difference(bore)

    def cut_countersink_below(plate, x, y, ctr_r, ctr_depth):
        """Countersink opens DOWNWARD from the plate's bottom face, so the screw
        head (visible only from below) sits flush with the plate's underside."""
        ctr = trimesh.creation.cylinder(
            radius=ctr_r, height=ctr_depth + 0.5, sections=32,
        )
        ctr.apply_translation([x, y, z_bot + ctr_depth / 2 - 0.25])
        return plate.difference(ctr)

    wedge_screw_holes = []
    for ang_deg in STRUT_ANGLES_DEG:
        ang = math.radians(ang_deg)
        tx, ty = -math.sin(ang), math.cos(ang)  # tangential unit vector
        for side in (+1, -1):
            x = TAP_HOLE_R * math.cos(ang) + tx * side * TAP_OFFSET
            y = TAP_HOLE_R * math.sin(ang) + ty * side * TAP_OFFSET
            plate = cut_through(plate, x, y, WEDGE_SCREW_CLEAR_R)
            plate = cut_countersink_below(plate, x, y,
                                          WEDGE_SCREW_COUNTER_R,
                                          WEDGE_SCREW_COUNTER_DEPTH)
            wedge_screw_holes.append({
                "x": x, "y": y, "angle_deg": ang_deg, "side": side,
            })
            print(f"    wedge-screw  @ {ang_deg:5.1f}° side {'+' if side > 0 else '-'}  "
                  f"({x:+7.2f}, {y:+7.2f})  M4 csk from below")

    # Export
    obj_path = out_dir / "bottom_plate.obj"
    stl_path = out_dir / "bottom_plate.stl"
    plate.export(obj_path)
    plate.export(stl_path)
    print(f"\nWrote {obj_path}  ({len(plate.vertices)} verts, {len(plate.faces)} faces)")
    print(f"Wrote {stl_path}")

    # Save the screw pattern so the drilling step can drill matching holes
    # in the drum skirt. Coords are in the printable STL (Z-up) system:
    # 'skirt_z' is the Z coord of the drum skirt bottom rim.
    pattern = {
        "skirt_z": args.skirt_y,           # NB: named 'skirt_y' historically; meaning is "vertical position"
        "skirt_y": args.skirt_y,           # legacy alias for back-compat
        "plate_thickness": args.thickness,
        "hole_circle_r": hole_circle_r,
        "n_holes": args.n_holes,
        "clear_r": args.hole_r_clear,
        "holes": holes,
    }
    pat_path = out_dir / "bottom_plate_screw_pattern.json"
    with open(pat_path, "w") as f:
        json.dump(pattern, f, indent=2)
    print(f"Wrote {pat_path}")


if __name__ == "__main__":
    main()
