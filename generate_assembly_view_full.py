#!/usr/bin/env python3
"""
Combine all printable parts into a single assembled view showing how the
whole pan fits together. Each part is placed at its installed position.

Parts:
    pan_printable_drilled.stl      — the pan body (all 7 pieces in position;
                                     same file as before split, with all
                                     bolt holes, dowel holes, bosses)
    bottom_plate.stl               — flat plate at the skirt base
    locking_ring.stl               — placed inside cavity at Z=-50 (bottom
                                     of the mounting bosses)
    dowel_pin.stl × 6              — one at each R=110 pin hole position

Output:
    pipeline_output/assembly_view.stl
    pipeline_output/assembly_view_exploded.stl  (parts shifted for clarity)
"""
from pathlib import Path
import math
import numpy as np
import trimesh

OUT_DIR = Path("pipeline_output")
PAN = OUT_DIR / "pan_printable_drilled.stl"
BOTTOM = OUT_DIR / "bottom_plate.stl"
RING = OUT_DIR / "locking_ring.stl"
DOWEL = OUT_DIR / "dowel_pin.stl"

# Locations matching add_assembly_holes.py and generate_pin_and_ring.py
SECTOR_MID_ANGLES_DEG = [74.5, 108.5, 167.0, 227.5, 259.5, 340.0]
CYL_SPLIT_R = 110.0
PIN_HOLE_Z = -30.0          # center of the radial pin holes
LOCKING_RING_Z = -50.0      # bottom of mounting bosses → ring sits here


def load(p):
    if not p.exists():
        print(f"  WARN missing {p}")
        return None
    return trimesh.load(p)


def position_locking_ring(ring):
    """Ring is generated centered at origin (XY in plane, Z=±thickness/2).
    Place its top face at Z = LOCKING_RING_Z. Ring already in correct
    XY plane orientation (axis along Z)."""
    # Determine current Z center
    cz = (ring.bounds[0, 2] + ring.bounds[1, 2]) / 2
    # Move so the TOP face is at LOCKING_RING_Z
    z_top_current = ring.bounds[1, 2]
    shift_z = LOCKING_RING_Z - z_top_current
    ring.apply_translation([0, 0, shift_z])
    return ring


def position_dowel_pin(pin, angle_deg, R, Z):
    """Dowel pin generated as a Z-axis cylinder centered at origin.
    We need it horizontal, axis along the RADIAL direction at angle_deg,
    centered at (R*cos, R*sin, Z)."""
    # Rotate cylinder so its axis goes from +Z to point radially.
    # Radial direction at angle_deg in XY plane = (cos a, sin a, 0).
    # We need a rotation that takes (0,0,1) to (cos a, sin a, 0).
    a = math.radians(angle_deg)
    target = np.array([math.cos(a), math.sin(a), 0])
    # Rotation axis = cross( (0,0,1), target ) = (-sin a, cos a, 0)
    rot_axis = np.array([-math.sin(a), math.cos(a), 0])
    R_mat = trimesh.transformations.rotation_matrix(
        math.pi / 2, rot_axis,
    )
    pin = pin.copy()
    pin.apply_transform(R_mat)
    pin.apply_translation([R * math.cos(a), R * math.sin(a), Z])
    return pin


print(f"Loading parts...")
pan = load(PAN)
bottom = load(BOTTOM)
ring = load(RING)
dowel = load(DOWEL)

print(f"  pan:    {len(pan.faces):,} tris, bbox = {pan.bounds[1] - pan.bounds[0]}")
print(f"  bottom: {len(bottom.faces):,} tris, bbox = {bottom.bounds[1] - bottom.bounds[0]}")
print(f"  ring:   {len(ring.faces):,} tris, bbox = {ring.bounds[1] - ring.bounds[0]}")
print(f"  dowel:  {len(dowel.faces):,} tris, bbox = {dowel.bounds[1] - dowel.bounds[0]}")

# Position the locking ring
ring = position_locking_ring(ring)

# Place 6 dowel pins, one at each R=110 sector midpoint
dowels = []
for ang_deg in SECTOR_MID_ANGLES_DEG:
    pin = position_dowel_pin(dowel, ang_deg, CYL_SPLIT_R, PIN_HOLE_Z)
    dowels.append(pin)

# ─────────────────────────────────────────────────────────────────────────
# Assembled view
# ─────────────────────────────────────────────────────────────────────────
assembled = trimesh.util.concatenate([pan, bottom, ring] + dowels)
out_a = OUT_DIR / "assembly_view.stl"
assembled.export(out_a)
print(f"\nAssembled: {len(assembled.faces):,} tris -> {out_a}")

# ─────────────────────────────────────────────────────────────────────────
# Exploded view: shift parts along Z so you can see them separately
# ─────────────────────────────────────────────────────────────────────────
print("\nExploded view (parts shifted along Z)...")
pan_e = pan.copy()
# Pan stays at original position
bottom_e = bottom.copy()
bottom_e.apply_translation([0, 0, -80])      # 80mm below skirt
ring_e = ring.copy()
ring_e.apply_translation([0, 0, -40])        # 40mm below installed position
dowels_e = []
for i, pin in enumerate(dowels):
    p = pin.copy()
    # Move dowel outward in its own radial direction
    ang = math.radians(SECTOR_MID_ANGLES_DEG[i])
    dx = 80 * math.cos(ang)
    dy = 80 * math.sin(ang)
    p.apply_translation([dx, dy, 0])
    dowels_e.append(p)

exploded = trimesh.util.concatenate([pan_e, bottom_e, ring_e] + dowels_e)
out_e = OUT_DIR / "assembly_view_exploded.stl"
exploded.export(out_e)
print(f"  Exploded: {len(exploded.faces):,} tris -> {out_e}")
