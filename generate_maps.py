#!/usr/bin/env python3
"""Generate map images showing pads, symmetric hardware masks, grooves, and screw holes."""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon as ShapelyPolygon
import sys

sys.path.insert(0, '.')
from generate_notepad import (
    parse_obj_file, extract_object_mesh, compute_surface_normal,
    compute_interior_centroid, compute_leveling_rotation,
    find_boundary_edges, find_all_boundary_loops,
    NOTE_BY_INDEX, CM_TO_MM, PAN_SCALE, PAN_THICKNESS, check_and_scale_pad,
    MIN_PAD_SIZE, CENTRAL_MIN_PAD_SIZE,
)

OBJ_PATH = "data/Tenor Pan only.obj"
HW_BUFFER = 2.5  # must match generate_pan_body.py


def load_properties():
    with open("data/notepads/notepad_properties.json") as f:
        return {p['index']: p for p in json.load(f)}


def project_to_xz(verts, R_level, offset):
    """Project vertices to leveled XZ plane."""
    body = (R_level @ (verts - offset).T).T
    return body[:, 0], body[:, 2]  # X, Z in body coords


def compute_symmetric_hw_mask(hw_xz_points, pad_xz_points):
    """Compute symmetric hardware mask polygon in XZ coords.

    Mirrors the hardware convex hull about the pad's PCA long axis,
    unions, clips to pad boundary, and buffers.

    Returns the buffered, pad-clipped, symmetric mask as a Shapely Polygon,
    or None if empty.
    """
    hull = ConvexHull(hw_xz_points)
    hw_poly = ShapelyPolygon(hw_xz_points[hull.vertices])

    # Pad long axis via PCA
    pad_centroid_xz = pad_xz_points.mean(axis=0)
    centered = pad_xz_points - pad_centroid_xz
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    long_axis = eigvecs[:, np.argmax(eigvals)]

    # Reflect hw_poly about the line through pad centroid along long_axis
    coords = np.array(hw_poly.exterior.coords)
    rel = coords - pad_centroid_xz
    proj = np.outer(rel @ long_axis, long_axis)
    reflected = pad_centroid_xz + 2 * proj - rel
    mirror_poly = ShapelyPolygon(reflected)
    hw_poly = hw_poly.union(mirror_poly)
    if hw_poly.geom_type == 'MultiPolygon':
        hw_poly = hw_poly.convex_hull

    pad_hull = ConvexHull(pad_xz_points)
    pad_poly = ShapelyPolygon(pad_xz_points[pad_hull.vertices])
    clipped = pad_poly.intersection(hw_poly)

    if clipped.is_empty:
        return None

    buffered = clipped.buffer(HW_BUFFER)
    buffered = pad_poly.intersection(buffered)

    if buffered.geom_type == 'Polygon' and not buffered.is_empty:
        return buffered
    return None


def load_assembly_view():
    """Load assembly_view.obj and return (verts_array, objects_dict)."""
    verts = []
    objects = {}
    current = None
    with open('data/notepads/assembly_view.obj') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith('o '):
                current = line[2:]
                objects[current] = set()
            elif line.startswith('f ') and current:
                for p in line.split()[1:]:
                    try:
                        objects[current].add(int(p.split('/')[0]) - 1)
                    except ValueError:
                        pass
    return np.array(verts), objects


def main():
    props = load_properties()
    objects, all_vertices = parse_obj_file(OBJ_PATH)

    offset_path = Path("data/pan_centroid_offset.json")
    with open(offset_path) as f:
        od = json.load(f)
    pan_offset = np.array(od["centroid_offset_mm"])
    R_level = compute_leveling_rotation(OBJ_PATH)

    # Load assembly view for hardware vertices in body coords
    a_verts, a_objects = load_assembly_view()

    ring_colors = {'outer': '#ffcccc', 'central': '#ccffcc', 'inner': '#ccccff'}
    ring_edge = {'outer': '#ff6666', 'central': '#66cc66', 'inner': '#6666ff'}
    hw_fill = {'outer': '#ff9999', 'central': '#99ff99', 'inner': '#9999ff'}
    groove_line = {'outer': '#cc4444', 'central': '#44aa44', 'inner': '#4444cc'}

    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 12))
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 12))
    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 12))

    for idx in sorted(NOTE_BY_INDEX.keys()):
        info = NOTE_BY_INDEX[idx]
        ring = info['ring']
        p = props.get(idx)
        if not p:
            continue

        # Extract pad + groove geometry
        min_size = MIN_PAD_SIZE if ring == 'outer' else CENTRAL_MIN_PAD_SIZE
        pv, pf = extract_object_mesh(objects, info['pan_object'], all_vertices)
        gv, gf = extract_object_mesh(objects, info['grove_object'], all_vertices)
        pv, gv, _, _ = check_and_scale_pad(pv, gv, min_size=min_size)

        # Pad boundary in body XZ
        be = find_boundary_edges(pf)
        loops = find_all_boundary_loops(be)
        if not loops:
            continue
        bvi = max(loops, key=len)
        bpts = pv[bvi]
        bx, bz = project_to_xz(bpts, R_level, pan_offset)
        bx = np.append(bx, bx[0])
        bz = np.append(bz, bz[0])

        # Groove boundary in body XZ
        gbe = find_boundary_edges(gf)
        gloops = find_all_boundary_loops(gbe)
        grove_bx, grove_bz = None, None
        if gloops:
            gvi = max(gloops, key=len)
            gbpts = gv[gvi]
            grove_bx, grove_bz = project_to_xz(gbpts, R_level, pan_offset)
            grove_bx = np.append(grove_bx, grove_bx[0])
            grove_bz = np.append(grove_bz, grove_bz[0])

        # Compute symmetric hardware mask from assembly view
        hw_mask_poly = None
        hw_xz_list = []
        for prefix in ['MountBase_', 'Sleeve_']:
            key = prefix + idx
            if key in a_objects:
                hv = a_verts[sorted(a_objects[key])]
                hw_xz_list.append(hv[:, [0, 2]])

        pad_key = f'Pad_{idx}'
        if hw_xz_list and pad_key in a_objects:
            all_hw_xz = np.vstack(hw_xz_list)
            pv_body = a_verts[sorted(a_objects[pad_key])]
            pad_xz_body = pv_body[:, [0, 2]]
            hw_mask_poly = compute_symmetric_hw_mask(all_hw_xz, pad_xz_body)

        # Mount center in body coords
        centroid = np.array(p['centroid'])
        cx, cz = project_to_xz(centroid.reshape(1, -1), R_level, pan_offset)
        cx, cz = cx[0], cz[0]

        # Screw holes
        hole_pts = [np.array(hp) for hp in p.get('hole_positions', [])]

        # --- Map 1: Symmetric hardware masks + groove outlines ---
        ax1.fill(bx, bz, color=ring_colors[ring], alpha=0.15,
                 edgecolor=ring_edge[ring], linewidth=0.5)
        if hw_mask_poly is not None:
            mc = np.array(hw_mask_poly.exterior.coords)
            ax1.fill(mc[:, 0], mc[:, 1], color=hw_fill[ring], alpha=0.4,
                     edgecolor=ring_edge[ring], linewidth=0.8)
        if grove_bx is not None:
            ax1.plot(grove_bx, grove_bz, color=groove_line[ring],
                     linewidth=0.8, linestyle='--')
        ax1.plot(cx, cz, '.', color=ring_edge[ring], markersize=2)

        # --- Map 2: Pads + grooves + screw holes ---
        ax2.fill(bx, bz, color=ring_colors[ring], alpha=0.3,
                 edgecolor=ring_edge[ring], linewidth=0.8)
        if grove_bx is not None:
            ax2.plot(grove_bx, grove_bz, color=groove_line[ring],
                     linewidth=0.8, linestyle='--')
        ax2.plot(cx, cz, '.', color=ring_edge[ring], markersize=2)
        for hp in hole_pts:
            hx, hz = project_to_xz(hp.reshape(1, -1), R_level, pan_offset)
            ax2.plot(hx[0], hz[0], 'k.', markersize=4)

        # --- Map 3: Overlay — mask + grooves + screw holes ---
        ax3.plot(bx, bz, color=ring_edge[ring], linewidth=0.8)
        if hw_mask_poly is not None:
            mc = np.array(hw_mask_poly.exterior.coords)
            ax3.fill(mc[:, 0], mc[:, 1], color='#ffdd44', alpha=0.6,
                     edgecolor='#886600', linewidth=0.8)
        if grove_bx is not None:
            ax3.plot(grove_bx, grove_bz, color='#88aacc',
                     linewidth=1.0, linestyle='--')
        ax3.plot(cx, cz, '.', color='#886600', markersize=2)
        for hp in hole_pts:
            hx, hz = project_to_xz(hp.reshape(1, -1), R_level, pan_offset)
            ax3.plot(hx[0], hz[0], 'kx', markersize=4, markeredgewidth=0.8)

    for ax, title, fname in [
        (ax1, 'MAP 1: Symmetric Hardware Masks + Grooves\n'
              'Filled=mask (symmetric, buffered), Dashed=groove boundary',
         'data/map1_hardware.png'),
        (ax2, 'MAP 2: Pads + Grooves + Screw Holes',
         'data/map2_pads.png'),
        (ax3, 'MAP 3: Mask Overlay + Grooves\n'
              'Yellow=symmetric mask | Dashed=groove | X=screw',
         'data/map3_overlay.png'),
    ]:
        ax.set_aspect('equal')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Z (mm)')
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.15)

    fig1.tight_layout()
    fig1.savefig('data/map1_hardware.png', dpi=150)
    plt.close(fig1)
    fig2.tight_layout()
    fig2.savefig('data/map2_pads.png', dpi=150)
    plt.close(fig2)
    fig3.tight_layout()
    fig3.savefig('data/map3_overlay.png', dpi=150)
    plt.close(fig3)
    print("Saved: data/map1_hardware.png, map2_pads.png, map3_overlay.png")


if __name__ == "__main__":
    main()
