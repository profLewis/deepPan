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
    p.add_argument("--skirt-y", type=float, default=-122.7,
                   help="Y of drum-skirt bottom rim (mm); plate top sits here")
    p.add_argument("--out-dir", default="pipeline_output")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    plate_r = args.skirt_r + args.plate_margin
    hole_circle_r = args.skirt_r - args.hole_circle_margin
    y_top = args.skirt_y
    y_bot = y_top - args.thickness

    print(f"Plate: outer R={plate_r}mm, thickness={args.thickness}mm")
    print(f"  Y top  = {y_top:.2f} (sits flush against drum skirt bottom)")
    print(f"  Y bot  = {y_bot:.2f}")
    print(f"  {args.n_holes} M4 countersunk holes at R = {hole_circle_r:.2f}mm")

    # Build plate solid as a cylinder via trimesh
    plate = trimesh.creation.cylinder(
        radius=plate_r,
        height=args.thickness,
        sections=128,
    )
    plate.apply_translation([0, (y_top + y_bot) / 2, 0])
    # trimesh creates cylinder along Z by default — rotate to Y axis
    R = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    plate.apply_transform(R)
    # Recenter (rotation about origin may shift)
    plate.apply_translation([0, (y_top + y_bot) / 2 - plate.bounds.mean(axis=0)[1], 0])

    # Hole positions
    holes = []
    for i in range(args.n_holes):
        ang = 2 * math.pi * i / args.n_holes
        # Stagger so holes don't sit on the strut planes (15+30i deg).
        # Strut angles are 15, 75, 135, 195, 255, 315 deg. Offset by 22.5 deg.
        ang += math.radians(22.5)
        x = hole_circle_r * math.cos(ang)
        z = hole_circle_r * math.sin(ang)
        holes.append({
            "x": x, "z": z,
            "angle_deg": math.degrees(ang),
            "clear_r": args.hole_r_clear,
            "counter_r": args.counter_r,
            "counter_depth": args.counter_depth,
        })

    # Subtract M4 through-holes + countersinks
    for h in holes:
        # Through-hole
        bore = trimesh.creation.cylinder(
            radius=h["clear_r"], height=args.thickness + 1.0, sections=32,
        )
        R2 = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        bore.apply_transform(R2)
        bore.apply_translation([h["x"], (y_top + y_bot) / 2, h["z"]])
        plate = plate.difference(bore)
        # Countersink (cone-ish; use a cylinder with stepped diameter for FDM-friendly)
        ctr = trimesh.creation.cylinder(
            radius=h["counter_r"],
            height=args.counter_depth + 0.5,
            sections=32,
        )
        ctr.apply_transform(R2)
        ctr_y = y_top - args.counter_depth / 2 + 0.25
        ctr.apply_translation([h["x"], ctr_y, h["z"]])
        plate = plate.difference(ctr)
        print(f"  hole at ({h['x']:+7.2f}, {h['z']:+7.2f})  "
              f"angle={h['angle_deg']:6.2f}deg")

    # Export
    obj_path = out_dir / "bottom_plate.obj"
    stl_path = out_dir / "bottom_plate.stl"
    plate.export(obj_path)
    plate.export(stl_path)
    print(f"\nWrote {obj_path}  ({len(plate.vertices)} verts, {len(plate.faces)} faces)")
    print(f"Wrote {stl_path}")

    # Save the screw pattern so the splitting step can drill matching holes
    pattern = {
        "skirt_y": args.skirt_y,
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
