#!/usr/bin/env python3
"""
Generate a multi-object OBJ assembly view of the full pan after splitting.

All parts kept as SEPARATE objects in the OBJ so each can be toggled in
Blender:
  * pan_piece_central                  (1)
  * pan_piece_outer_0 … pan_piece_outer_5  (6)
  * bottom_plate                       (1)
  * peg_strut<angle>_pos<n>            (12 HORIZONTAL alignment pegs)
  * mount_outer_base_O*                (12 outer-ring mount bases — NO sleeves)
  * mount_small_*                      (17 central/inner push-caps)

Each mount is placed at its pad centroid (from notepad_properties.json) with
its main axis aligned along the pad normal — extending INTO the cavity.

Output:
    pipeline_output/assembly_view_with_plugs.obj
    pipeline_output/assembly_view_with_plugs.stl
"""
from pathlib import Path
import json
import math
import numpy as np
import trimesh

OUT_DIR = Path("pipeline_output")
DATA_DIR = Path("data")

# Must match split_assembly.py
STRUT_ANGLES_DEG = [15.0, 75.0, 135.0, 195.0, 255.0, 315.0]
PEG_POSITIONS    = [(130.0, -55.0), (200.0, -90.0)]


def load(p):
    if not p.exists():
        print(f"  WARN missing {p}")
        return None
    m = trimesh.load(p, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(list(m.geometry.values()))
    return m


def place_peg_horizontal(plug_mesh, angle_deg, R, Z):
    """Plug cylinder along Z by default. Rotate so axis is TANGENTIAL at
    angle_deg, then translate to (R cos θ, R sin θ, Z)."""
    ang = math.radians(angle_deg)
    Rx = trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0])
    Rz = trimesh.transformations.rotation_matrix(ang, [0, 0, 1])
    p = plug_mesh.copy()
    p.apply_transform(Rx)
    p.apply_transform(Rz)
    p.apply_translation([R * math.cos(ang), R * math.sin(ang), Z])
    return p


# ── Pad-coordinate transform: raw scan (Y-up, mm) → pan_printable (Z-up, mm)
def load_pad_transform():
    offset = json.load(open(DATA_DIR / "pan_centroid_offset.json"))
    pan_centroid_offset = np.array(offset["centroid_offset_mm"])
    return pan_centroid_offset


def transform_point(pt, offset):
    """Raw scan (Y-up, mm) → pan_printable Z-up frame.
       1) subtract pan_centroid_offset; 2) rotate +90° around X.  In rotation
       (x, y, z) → (x, -z, y)."""
    p = np.array(pt) - offset
    return np.array([p[0], -p[2], p[1]])


def transform_vector(v):
    """Same rotation for direction vectors (no translation)."""
    v = np.array(v)
    return np.array([v[0], -v[2], v[1]])


def align_z_to(direction):
    """4x4 rotation matrix that maps the local +Z axis to `direction`."""
    z = np.array([0.0, 0.0, 1.0])
    d = direction / np.linalg.norm(direction)
    if np.allclose(d, z):
        return np.eye(4)
    if np.allclose(d, -z):
        return trimesh.transformations.rotation_matrix(math.pi, [1.0, 0.0, 0.0])
    axis = np.cross(z, d)
    axis = axis / np.linalg.norm(axis)
    angle = math.acos(float(np.clip(np.dot(z, d), -1.0, 1.0)))
    return trimesh.transformations.rotation_matrix(angle, axis)


def place_mount(mount_proto, pad, offset):
    """Position a mount mesh at a pad: centroid + normal-aligned axis.
    Mount's local +Z extends INTO the cavity (= -pad_normal in world)."""
    centroid_zup = transform_point(pad["centroid"], offset)
    normal_zup   = transform_vector(pad["normal"])
    target = -normal_zup  # mount extends into cavity
    R = align_z_to(target)
    m = mount_proto.copy()
    m.apply_transform(R)
    m.apply_translation(centroid_zup)
    return m


def main():
    parts = {}

    # Pan body pieces + bottom plate
    for name in ["pan_piece_central"] + [f"pan_piece_outer_{i}" for i in range(6)]:
        m = load(OUT_DIR / f"{name}.stl")
        if m is not None:
            parts[name] = m
            print(f"  {name}: {len(m.faces):,} tris")
    bp = load(OUT_DIR / "bottom_plate.stl")
    if bp is not None:
        parts["bottom_plate"] = bp
        print(f"  bottom_plate: {len(bp.faces):,} tris")

    # 12 horizontal pegs
    plug = load(OUT_DIR / "plug.stl")
    if plug is None:
        print("ERROR: plug.stl missing — run generate_plug.py first")
        return
    for ang_deg in STRUT_ANGLES_DEG:
        for j, (R, Z) in enumerate(PEG_POSITIONS):
            p = place_peg_horizontal(plug, ang_deg, R, Z)
            name = f"peg_strut{int(ang_deg):03d}_pos{j}"
            parts[name] = p

    # ── Mount placements (outer BASES only, no sleeves; small pushcaps)
    pad_props = json.load(open(DATA_DIR / "notepads" / "notepad_properties.json"))
    offset = load_pad_transform()

    outer_base_src = OUT_DIR / "mount_large_base_outer_ring.stl"
    small_cap_src  = OUT_DIR / "mount_small_pushcap_central_inner.stl"

    outer_base_proto = load(outer_base_src)
    small_cap_proto  = load(small_cap_src)

    # Decimate the outer base (1.4M tris is too heavy for an assembly view)
    if outer_base_proto is not None and len(outer_base_proto.faces) > 60000:
        print(f"  Decimating outer-base proto ({len(outer_base_proto.faces):,} → ~50k tris)...")
        try:
            outer_base_proto = outer_base_proto.simplify_quadric_decimation(face_count=50000)
        except Exception as e:
            print(f"    decimation failed ({e}); using as-is")
    if outer_base_proto is not None:
        print(f"  outer-base proto: {len(outer_base_proto.faces):,} tris")
    if small_cap_proto is not None:
        print(f"  small-cap proto:  {len(small_cap_proto.faces):,} tris")

    n_outer = n_small = 0
    for pad in pad_props:
        ring = pad["ring"]
        idx = pad["index"]
        if ring == "outer" and outer_base_proto is not None:
            m = place_mount(outer_base_proto, pad, offset)
            parts[f"mount_outer_base_{idx}"] = m
            n_outer += 1
        elif ring in ("central", "inner") and small_cap_proto is not None:
            m = place_mount(small_cap_proto, pad, offset)
            parts[f"mount_small_{idx}"] = m
            n_small += 1
    print(f"  placed: {n_outer} outer bases (no sleeves), {n_small} small pushcaps")

    # ── Export multi-object OBJ ──
    obj_path = OUT_DIR / "assembly_view_with_plugs.obj"
    with open(obj_path, "w") as f:
        vert_offset = 0
        for name, m in parts.items():
            f.write(f"o {name}\n")
            for v in m.vertices:
                f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            for face in m.faces:
                a, b, c = (face[0] + 1 + vert_offset,
                           face[1] + 1 + vert_offset,
                           face[2] + 1 + vert_offset)
                f.write(f"f {a} {b} {c}\n")
            vert_offset += len(m.vertices)
    print(f"\nWrote {obj_path} ({len(parts)} separate objects)")

    # ── Combined STL for quick view ──
    stl_path = OUT_DIR / "assembly_view_with_plugs.stl"
    combined = trimesh.util.concatenate(list(parts.values()))
    combined.export(stl_path)
    print(f"Wrote {stl_path} ({len(combined.faces):,} tris combined)")


if __name__ == "__main__":
    main()
