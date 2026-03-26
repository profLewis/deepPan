#!/usr/bin/env python3
"""
Generate pan_surface_up_shell.obj — pan_surface_up.obj offset to pad base
with hardware clearance holes carefully carved within each pad's boundary.

Holes are cut only within each pad's outline (with buffer), never beyond it.
"""

import numpy as np
import json
import math

from generate_notepad import (
    parse_obj_file, extract_object_mesh, compute_surface_normal,
    compute_interior_centroid, find_boundary_edges, find_all_boundary_loops,
    _point_in_polygon_2d,
    NOTE_BY_INDEX, CM_TO_MM, PAN_SCALE, PAN_THICKNESS,
    CENTRAL_MOUNT_INNER_DIAMETER, CENTRAL_MOUNT_WALL_THICKNESS,
    MOUNT_INNER_DIAMETER, MOUNT_WALL_THICKNESS, MOUNT_THREAD_DEPTH,
    SENSOR_SKIN,
    check_and_scale_pad, CENTRAL_MIN_PAD_SIZE, MIN_PAD_SIZE,
    compute_leveling_rotation,
)
from generate_outer_sleeve import SLEEVE_INNER_DIAMETER, SLEEVE_WALL_THICKNESS

INPUT_PATH = "data/pan_surface_up.obj"
OUTPUT_PATH = "data/pan_surface_up_shell.obj"

# Clearance beyond hardware edge
HOLE_TOLERANCE = 2.0    # mm beyond hardware radius
# Buffer inside pad boundary (faces within this distance of pad edge are kept)
PAD_EDGE_BUFFER = 2.0   # mm inward from pad boundary — don't carve here


def load_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '):
                face = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
                faces.append(face)
    return np.array(verts), faces


def compute_vertex_normals(verts, faces):
    normals = np.zeros_like(verts)
    for face in faces:
        if len(face) < 3:
            continue
        v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
        fn = np.cross(v1 - v0, v2 - v0)
        fn_len = np.linalg.norm(fn)
        if fn_len > 1e-12:
            fn /= fn_len
        for vi in face:
            normals[vi] += fn
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    lens = np.where(lens < 1e-12, 1.0, lens)
    normals /= lens
    return normals


def min_dist_to_polygon(px, py, poly_x, poly_y):
    """Minimum distance from point to polygon boundary."""
    n = len(poly_x)
    min_d = float('inf')
    for i in range(n):
        j = (i + 1) % n
        ax, ay = poly_x[j] - poly_x[i], poly_y[j] - poly_y[i]
        bx, by = px - poly_x[i], py - poly_y[i]
        seg_len_sq = ax * ax + ay * ay
        if seg_len_sq < 1e-12:
            d_sq = bx * bx + by * by
        else:
            t = max(0, min(1, (bx * ax + by * ay) / seg_len_sq))
            dx, dy = bx - t * ax, by - t * ay
            d_sq = dx * dx + dy * dy
        if d_sq < min_d:
            min_d = d_sq
    return math.sqrt(min_d)


def main():
    print("Generating carefully carved pan shell...")

    verts, faces = load_obj(INPUT_PATH)
    print(f"  Input: {len(verts)} verts, {len(faces)} faces")

    # Offset along normals
    vnormals = compute_vertex_normals(verts, faces)
    shell_verts = verts - vnormals * PAN_THICKNESS
    print(f"  Offset {PAN_THICKNESS}mm along normals")

    # Load transforms
    with open("data/pan_centroid_offset.json") as f:
        pan_offset = np.array(json.load(f)["centroid_offset_mm"])
    R_level = compute_leveling_rotation("data/Tenor Pan only.obj")
    objects, all_vertices_raw = parse_obj_file("data/Tenor Pan only.obj")

    # Build per-pad data: boundary polygon in body coords + mount clearance
    pads = []
    for idx in sorted(NOTE_BY_INDEX.keys()):
        info = NOTE_BY_INDEX[idx]
        ring = info['ring']

        pv, pf = extract_object_mesh(objects, info['pan_object'], all_vertices_raw)
        gv, gf = extract_object_mesh(objects, info['grove_object'], all_vertices_raw)

        if ring == 'outer':
            pv, gv, _, _ = check_and_scale_pad(pv, gv, min_size=MIN_PAD_SIZE)
        else:
            pv, gv, _, _ = check_and_scale_pad(pv, gv, min_size=CENTRAL_MIN_PAD_SIZE)

        pan_normal = compute_surface_normal(pv, pf)
        _, centroid_mm = compute_interior_centroid(pv, pf, pan_normal, PAN_THICKNESS, 0)

        # Outer ring: boundary-loop centroid
        if ring == 'outer':
            be = find_boundary_edges(pf)
            loops = find_all_boundary_loops(be)
            if loops:
                bvi = max(loops, key=len)
                bc = pv[bvi].mean(axis=0)
                diff = bc - centroid_mm
                centroid_mm = bc - np.dot(diff, pan_normal) * pan_normal

        # Mount center in body coords
        center_body = R_level @ (centroid_mm - pan_offset)

        # Hardware clearance radius
        if ring == 'outer':
            hw_r = (SLEEVE_INNER_DIAMETER / 2 + SLEEVE_WALL_THICKNESS) + HOLE_TOLERANCE
        else:
            hw_r = (CENTRAL_MOUNT_INNER_DIAMETER / 2 + CENTRAL_MOUNT_WALL_THICKNESS) + HOLE_TOLERANCE

        # Pad boundary in body coords (for containment check)
        be = find_boundary_edges(pf)
        loops = find_all_boundary_loops(be)
        if not loops:
            continue
        bvi = max(loops, key=len)
        bpts_mm = pv[bvi]
        bpts_body = (R_level @ (bpts_mm - pan_offset).T).T

        # Build 2D projection of boundary onto tangent plane at mount center
        n_hat = R_level @ pan_normal
        n_hat /= np.linalg.norm(n_hat)
        if abs(n_hat[0]) < 0.9:
            xl = np.cross(n_hat, [1, 0, 0])
        else:
            xl = np.cross(n_hat, [0, 1, 0])
        xl /= np.linalg.norm(xl)
        yl = np.cross(n_hat, xl)

        rel = bpts_body - center_body
        bx_2d = rel @ xl
        by_2d = rel @ yl

        pads.append({
            'idx': idx,
            'ring': ring,
            'center': center_body,
            'hw_r': hw_r,
            'normal': n_hat,
            'xl': xl,
            'yl': yl,
            'bx': bx_2d,
            'by': by_2d,
        })

    print(f"  {len(pads)} pads loaded with boundaries")

    # For each face in the shell, check if it should be removed:
    # Remove if:
    #   1. Face centroid is INSIDE a pad boundary (with edge buffer)
    #   2. Face centroid is WITHIN the hardware clearance radius of that pad's mount
    kept = []
    removed = 0
    for face in faces:
        fc = shell_verts[face].mean(axis=0)

        should_remove = False
        for pad in pads:
            # Project face centroid onto pad's tangent plane
            rel = fc - pad['center']
            fx = np.dot(rel, pad['xl'])
            fy = np.dot(rel, pad['yl'])

            # Check if inside pad boundary
            if not _point_in_polygon_2d(fx, fy, pad['bx'], pad['by']):
                continue

            # Check distance from pad edge — don't carve near the edge
            edge_dist = min_dist_to_polygon(fx, fy, pad['bx'], pad['by'])
            if edge_dist < PAD_EDGE_BUFFER:
                continue

            # Check if within hardware clearance radius
            dist_from_mount = math.sqrt(fx * fx + fy * fy)
            if dist_from_mount < pad['hw_r']:
                should_remove = True
                break

        if should_remove:
            removed += 1
        else:
            kept.append(face)

    print(f"  Removed {removed} faces, kept {len(kept)}")

    with open(OUTPUT_PATH, 'w') as f:
        f.write("# Pan surface shell — offset to pad base, hardware holes carved\n")
        f.write(f"# Holes only within pad boundaries, {PAD_EDGE_BUFFER}mm edge buffer\n")
        f.write(f"# Hardware tolerance: {HOLE_TOLERANCE}mm\n\n")
        for v in shell_verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in kept:
            f.write(f"f {' '.join(str(vi + 1) for vi in face)}\n")

    print(f"  Saved: {OUTPUT_PATH} ({len(shell_verts)} verts, {len(kept)} faces)")


if __name__ == "__main__":
    main()
