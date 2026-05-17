#!/usr/bin/env python3
"""
Generate 6 bottom-plate pie sectors directly so each sector fits the
Bambu P1S (256 × 256 × 256) build volume.

Each sector spans 60° (between adjacent strut planes 15/75/135/195/255/315°)
with its share of:
  * perimeter screw holes (M4 countersunk into drum skirt)
  * wedge-screw clearance holes (M4 countersunk into wedge bases)

Outputs:
    pipeline_output/bottom_plate_sector_0.stl … bottom_plate_sector_5.stl
"""
from pathlib import Path
import json
import math
import numpy as np
import trimesh

OUT_DIR = Path("pipeline_output")

# Plate geometry — must match generate_bottom_plate.py
SKIRT_R = 246.0
PLATE_MARGIN = 8.0
PLATE_R = SKIRT_R + PLATE_MARGIN   # 254 mm
THICKNESS = 6.0
SKIRT_Z = -122.0
Z_TOP = SKIRT_Z
Z_BOT = Z_TOP - THICKNESS
Z_CENTER = (Z_TOP + Z_BOT) / 2

# Perimeter screw pattern (matches generate_bottom_plate.py)
N_PERIM = 8
PERIM_R = SKIRT_R - 10.0           # 236
PERIM_CLEAR_R = 2.2
PERIM_COUNTER_R = 4.2
PERIM_COUNTER_DEPTH = 2.5

# Strut joinery (matches split_assembly.py + generate_bottom_plate.py)
STRUT_ANGLES_DEG = [15.0, 75.0, 135.0, 195.0, 255.0, 315.0]
TAP_HOLE_R = 170.0
TAP_OFFSET = 5.0
WEDGE_CLEAR_R = 2.25
WEDGE_COUNTER_R = 4.2
WEDGE_COUNTER_DEPTH = 2.5

P1S = 256.0


def sector_solid(a_start, a_end, r_outer, z_top, z_bot):
    """Build a pie sector solid by extruding the 2D pie polygon."""
    arc_span = a_end - a_start
    n_arc = max(32, int(arc_span * 2))
    pts = [(0.0, 0.0)]
    for i in range(n_arc + 1):
        t = math.radians(a_start + arc_span * i / n_arc)
        pts.append((r_outer * math.cos(t), r_outer * math.sin(t)))
    from shapely.geometry import Polygon
    poly = Polygon(pts)
    s = trimesh.creation.extrude_polygon(poly, height=z_top - z_bot)
    s.apply_translation([0.0, 0.0, z_bot])
    return s


def angle_in_sector(ang_deg, a0, a1):
    """Test whether angle (deg) is in [a0, a1] modulo 360."""
    if a1 > a0:
        return a0 <= (ang_deg % 360) < a1
    # wrap-around case (a1 < a0)
    return (ang_deg % 360) >= a0 or (ang_deg % 360) < a1


def cut_through_hole(mesh, x, y, r):
    bore = trimesh.creation.cylinder(
        radius=r, height=THICKNESS + 2.0, sections=32,
    )
    bore.apply_translation([x, y, Z_CENTER])
    return mesh.difference(bore)


def cut_countersink_below(mesh, x, y, ctr_r, ctr_depth):
    """Countersink opens DOWNWARD from the plate bottom face."""
    ctr = trimesh.creation.cylinder(
        radius=ctr_r, height=ctr_depth + 0.5, sections=32,
    )
    ctr.apply_translation([x, y, Z_BOT + ctr_depth / 2 - 0.25])
    return mesh.difference(ctr)


def main():
    # Pre-compute hole positions (so we can dispatch them per-sector)
    perim_holes = []
    for i in range(N_PERIM):
        ang = 2 * math.pi * i / N_PERIM + math.radians(22.5)
        x = PERIM_R * math.cos(ang)
        y = PERIM_R * math.sin(ang)
        perim_holes.append({"x": x, "y": y, "angle_deg": math.degrees(ang) % 360})

    wedge_holes = []
    for ang_deg in STRUT_ANGLES_DEG:
        ang = math.radians(ang_deg)
        tx, ty = -math.sin(ang), math.cos(ang)
        for side in (+1, -1):
            x = TAP_HOLE_R * math.cos(ang) + tx * side * TAP_OFFSET
            y = TAP_HOLE_R * math.sin(ang) + ty * side * TAP_OFFSET
            wedge_holes.append({"x": x, "y": y, "angle_deg": math.degrees(math.atan2(y, x)) % 360})

    n = len(STRUT_ANGLES_DEG)
    for i in range(n):
        a0 = STRUT_ANGLES_DEG[i]
        a1 = STRUT_ANGLES_DEG[(i + 1) % n]
        if a1 <= a0:
            a1 += 360.0
        print(f"\n— Sector {i} ({a0:.0f}..{a1:.0f}°) —")

        # Solid pie sector (no holes yet)
        sec = sector_solid(a0, a1, PLATE_R, Z_TOP, Z_BOT)
        print(f"  pie solid: {len(sec.faces):,} tris, is_volume={sec.is_volume}")

        # Drill perimeter holes that fall in this sector
        for h in perim_holes:
            if angle_in_sector(h["angle_deg"], a0, a1):
                sec = cut_through_hole(sec, h["x"], h["y"], PERIM_CLEAR_R)
                sec = cut_countersink_below(
                    sec, h["x"], h["y"], PERIM_COUNTER_R, PERIM_COUNTER_DEPTH)
                print(f"    perim hole at ({h['x']:+7.2f}, {h['y']:+7.2f})")

        # Drill wedge-screw holes that fall in this sector
        for h in wedge_holes:
            if angle_in_sector(h["angle_deg"], a0, a1):
                sec = cut_through_hole(sec, h["x"], h["y"], WEDGE_CLEAR_R)
                sec = cut_countersink_below(
                    sec, h["x"], h["y"], WEDGE_COUNTER_R, WEDGE_COUNTER_DEPTH)
                print(f"    wedge hole at ({h['x']:+7.2f}, {h['y']:+7.2f})")

        b = sec.bounds[1] - sec.bounds[0]
        fit = "OK" if max(b[0], b[1]) <= P1S else "TOO BIG"
        print(f"  bbox {b[0]:.1f} × {b[1]:.1f} × {b[2]:.1f} mm [{fit}]")
        out = OUT_DIR / f"bottom_plate_sector_{i}.stl"
        sec.export(out)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
