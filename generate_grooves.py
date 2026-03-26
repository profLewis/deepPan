#!/usr/bin/env python3
"""
Generate groove geometry with taper and height variation for all pads.

Groove taper profile:
  - Inner edge (at pad boundary + tolerance): flush with pad top
    (STEP_INNER below playing surface)
  - Outer edge (groove outer perimeter): flush with drum surface
    (0mm drop — smooth join with surrounding pan body)
  - Smooth transition between inner and outer edges
  - Small tolerance gap between groove and pad pocket

Output files (one per ring, each containing named objects):
  data/grooves/grooves_outer.obj   — 12 outer-ring grooves
  data/grooves/grooves_central.obj — 12 central-ring grooves
  data/grooves/grooves_inner.obj   — 5 inner-ring grooves
"""

import numpy as np
import json
from pathlib import Path
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point

from generate_notepad import (
    parse_obj_file, extract_object_mesh, compute_surface_normal,
    compute_interior_centroid, compute_leveling_rotation,
    find_boundary_edges, find_all_boundary_loops,
    check_and_scale_pad,
    NOTE_BY_INDEX, CM_TO_MM, PAN_SCALE, PAN_THICKNESS,
    MIN_PAD_SIZE, CENTRAL_MIN_PAD_SIZE,
)

OBJ_PATH = "data/Tenor Pan only.obj"
OUTPUT_DIR = Path("data/grooves")

# Taper parameters
STEP_INNER = 0.1                     # mm — groove inner edge drop (flush with pad top)
GROOVE_PAD_TOLERANCE = 0.5           # mm — gap between pad boundary and groove channel
GROOVE_THICKNESS = PAN_THICKNESS     # extrude depth matches pad thickness
GROOVE_RAISE = 0.25                  # mm — raise groove upward from drum surface


def process_groove(pan_verts, pan_faces, grove_verts, grove_faces):
    """Apply taper to groove and thicken into a solid mesh.

    Taper profile:
      - Inside pad boundary or within GROOVE_PAD_TOLERANCE: no drop (rim)
      - Inner edge (just beyond tolerance): drop = STEP_INNER (flush with pad top)
      - Outer edge (groove perimeter): drop = 0 (flush with drum surface)

    Returns (solid_verts, solid_faces) or None if groove is empty.
    """
    if len(grove_verts) < 3 or len(grove_faces) < 1:
        return None

    pan_normal = compute_surface_normal(pan_verts, pan_faces)
    n_hat = pan_normal / np.linalg.norm(pan_normal)
    _, pan_surface_centroid = compute_interior_centroid(
        pan_verts, pan_faces, pan_normal, PAN_THICKNESS, 0)

    # Pad boundary polygon in XZ for distance checks
    pan_xz = pan_verts[:, [0, 2]]
    try:
        ph = ConvexHull(pan_xz)
        pad_poly = Polygon(pan_xz[ph.vertices])
    except Exception:
        return None

    # Max distance any groove vertex is from the pad boundary
    grove_verts = grove_verts.copy()
    grove_max_r = 0.0
    for i in range(len(grove_verts)):
        pt = Point(grove_verts[i, 0], grove_verts[i, 2])
        if not pad_poly.contains(pt):
            grove_max_r = max(grove_max_r, pad_poly.exterior.distance(pt))

    effective_range = max(grove_max_r - GROOVE_PAD_TOLERANCE, 0.1)

    # Apply taper: inner edge (STEP_INNER) → outer edge (0)
    for i in range(len(grove_verts)):
        pt = Point(grove_verts[i, 0], grove_verts[i, 2])
        if pad_poly.contains(pt):
            # Inside pad boundary — keep at pad surface level (no carve here)
            pass
        else:
            d = pad_poly.exterior.distance(pt)
            if d < GROOVE_PAD_TOLERANCE:
                # Within tolerance gap — keep at surface level (creates rim)
                pass
            else:
                d_eff = d - GROOVE_PAD_TOLERANCE
                frac = min(d_eff / effective_range, 1.0)
                # Inner edge (frac=0): drop = STEP_INNER (flush with pad top)
                # Outer edge (frac=1): drop = 0 (flush with drum surface)
                drop = STEP_INNER * (1.0 - frac)
                grove_verts[i] -= n_hat * drop

    # Raise entire groove surface upward along normal
    grove_verts += n_hat * GROOVE_RAISE

    # Thicken: extrude straight down to flat bottom at pad-bottom plane
    surf_center = pan_verts.mean(axis=0)
    pad_bot_plane_d = np.dot(surf_center - n_hat * GROOVE_THICKNESS, n_hat)

    n_gv = len(grove_verts)
    grove_bot = np.zeros_like(grove_verts)
    for i in range(n_gv):
        v_d = np.dot(grove_verts[i], n_hat)
        grove_bot[i] = grove_verts[i] - n_hat * (v_d - pad_bot_plane_d)

    solid_verts = np.vstack([grove_verts, grove_bot])

    # Top faces (original winding)
    solid_faces = [list(f) for f in grove_faces]
    # Bottom faces (reversed winding)
    solid_faces += [[vi + n_gv for vi in reversed(f)] for f in grove_faces]

    # Side walls along all boundary loops
    gbe = find_boundary_edges(grove_faces)
    gloops = find_all_boundary_loops(gbe)
    for loop in gloops:
        for li in range(len(loop)):
            v1 = loop[li]
            v2 = loop[(li + 1) % len(loop)]
            solid_faces.append([v1, v2, v2 + n_gv, v1 + n_gv])

    return solid_verts, solid_faces


def main():
    print("=" * 60)
    print("Generating tapered grooves for all pads")
    print(f"  STEP_INNER={STEP_INNER}mm, tolerance={GROOVE_PAD_TOLERANCE}mm")
    print("  Inner edge: flush with pad top | Outer edge: flush with drum surface")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    objects, all_vertices = parse_obj_file(OBJ_PATH)
    R_level = compute_leveling_rotation(OBJ_PATH)

    offset_path = Path("data/pan_centroid_offset.json")
    with open(offset_path) as f:
        pan_offset = np.array(json.load(f)["centroid_offset_mm"])

    # Process all pads, group by ring
    ring_meshes = {'outer': [], 'central': [], 'inner': []}

    for idx in sorted(NOTE_BY_INDEX.keys()):
        info = NOTE_BY_INDEX[idx]
        ring = info['ring']

        pv, pf = extract_object_mesh(objects, info['pan_object'], all_vertices)
        gv, gf = extract_object_mesh(objects, info['grove_object'], all_vertices)

        min_size = MIN_PAD_SIZE if ring == 'outer' else CENTRAL_MIN_PAD_SIZE
        pv, gv, _, _ = check_and_scale_pad(pv, gv, min_size=min_size)

        result = process_groove(pv, pf, gv, gf)
        if result is None:
            print(f"  {idx}: no groove geometry — skipped")
            continue

        solid_verts, solid_faces = result

        # Transform to body coords (centered + leveled)
        body_verts = (R_level @ (solid_verts - pan_offset).T).T

        ring_meshes[ring].append((idx, body_verts, solid_faces))
        print(f"  {idx} ({info['note']}{info['octave']}): "
              f"{len(solid_verts)}v {len(solid_faces)}f")

    # Save per-ring OBJ files
    for ring in ['outer', 'central', 'inner']:
        meshes = ring_meshes[ring]
        if not meshes:
            continue
        out_path = OUTPUT_DIR / f"grooves_{ring}.obj"
        with open(out_path, 'w') as f:
            f.write(f"# Tapered grooves — {ring} ring\n")
            f.write(f"# Units: mm, body coords (matches pan_body.obj)\n\n")
            vert_offset = 0
            for note_idx, verts, faces in meshes:
                f.write(f"o Groove_{note_idx}\n")
                for v in verts:
                    f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
                for face in faces:
                    idx_str = ' '.join(str(vi + 1 + vert_offset) for vi in face)
                    f.write(f"f {idx_str}\n")
                vert_offset += len(verts)
            f.write("\n")
        print(f"\nSaved {out_path}  ({len(meshes)} grooves)")

    print("\nDone!")


if __name__ == "__main__":
    main()
