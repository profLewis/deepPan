#!/usr/bin/env python3
"""
Generate thickened pan surface (pan_surface_up_thick) with:
- 25mm shell thickness (pan_surface_up is the outer boundary)
- Hardware holes cut through full thickness (mount base for outer, boss for central/inner)
- Pad pockets sunk into the surface (pads sit flush with outer surface)
- M2 through-holes matching pad screw locations
- 15mm cylindrical base plate (bottom 15mm of drum replaced)
- M3x20 countersunk base attachment screws with attachment bosses
- Screw holes in the surface matching pad screw hole locations

Coordinates match pan_surface_up.obj and assembly_view.obj (R_level, Y=drum axis).

Usage:
    python generate_pan_surface_thick.py [--res=0.5]
"""

import numpy as np
import json
import struct
import math
import sys
import gc
from pathlib import Path
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from scipy.ndimage import zoom as ndimage_zoom
from skimage.measure import marching_cubes

from generate_sector import extract_bowl_surface
from generate_notepad import (
    compute_leveling_rotation, NOTE_BY_INDEX, NOTE_MAPPING, PAN_THICKNESS,
    MOUNT_INNER_DIAMETER, MOUNT_WALL_THICKNESS, MOUNT_THREAD_DEPTH,
    CENTRAL_MOUNT_INNER_DIAMETER, CENTRAL_MOUNT_WALL_THICKNESS,
    CENTRAL_MOUNT_THREAD_DEPTH,
)
from generate_mount_base import (
    BASE_INNER_DIAMETER as MOUNT_BASE_INNER_DIA,
    BASE_WALL_THICKNESS as MOUNT_BASE_WALL,
    EXT_THREAD_DEPTH as MOUNT_BASE_EXT_THREAD,
)
from generate_quarter import classify_drum_wall, reindex_mesh, subdivide_mesh

# Mount base outer radius (outer ring hardware — no sleeve)
MOUNT_BASE_OUTER_R = MOUNT_BASE_INNER_DIA / 2 + MOUNT_BASE_WALL + MOUNT_BASE_EXT_THREAD

# ============================================================
# Parameters
# ============================================================
SHELL_THICKNESS = 25.0        # mm — shell thickness everywhere
PAD_POCKET_TOLERANCE = 0.5    # mm — clearance below pad bottom
POCKET_DEPTH = PAN_THICKNESS + PAD_POCKET_TOLERANCE  # 5.5mm
BOSS_HOLE_TOLERANCE = 2.0     # mm — clearance around hardware outer radius
BASE_THICKNESS = 15.0         # mm — cylindrical base plate
SUBDIVIDE_ROUNDS = 2          # heightmap smoothness

# Pad surface screws (M2 through-holes matching notepad holes)
SURFACE_SCREW_R = 1.1         # mm — M2 clearance hole radius (2.2mm dia)

# Base attachment screws (M3x20 countersunk)
BASE_SCREW_N = 8              # number of screws around perimeter
BASE_SCREW_INSET = 12.0       # mm from drum outer edge
M3_CLEAR_R = 1.7              # mm — M3 clearance radius
M3_CSINK_R = 3.36             # mm — M3 90° countersink head radius
M3_CSINK_DEPTH = 2.0          # mm — countersink recess depth
M3_PILOT_R = 1.25             # mm — M3 tapped pilot radius
M3_PILOT_DEPTH = 5.0          # mm — thread engagement into attachment boss

# Attachment bosses (inside drum wall, receive base screws)
ATTACH_BOSS_R = 6.0           # mm — boss outer radius
ATTACH_BOSS_HEIGHT = 20.0     # mm — boss height above base plate top


def write_stl(path, verts, faces):
    """Write binary STL."""
    with open(path, 'wb') as f:
        f.write(b"STL pan_surface_thick".ljust(80, b'\0'))
        f.write(struct.pack('<I', len(faces)))
        for tri in faces:
            v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
            n = np.cross(v1 - v0, v2 - v0)
            nl = np.linalg.norm(n)
            if nl > 0:
                n /= nl
            f.write(struct.pack('<3f', *n))
            for v in (v0, v1, v2):
                f.write(struct.pack('<3f', *v))
            f.write(struct.pack('<H', 0))


def main():
    res = 0.5
    for arg in sys.argv[1:]:
        if arg.startswith('--res='):
            res = float(arg.split('=')[1])

    print("=" * 60)
    print(f"Pan Surface Thick — {SHELL_THICKNESS}mm shell, {res}mm voxel")
    print(f"  Pockets: {POCKET_DEPTH}mm deep, Boss tol: {BOSS_HOLE_TOLERANCE}mm")
    print(f"  Base: {BASE_THICKNESS}mm, {BASE_SCREW_N}x M3x20 countersunk")
    print("=" * 60)

    # --------------------------------------------------------
    # Phase 1: Load + level bowl surface
    # --------------------------------------------------------
    print("\nPhase 1: Load + level...")
    bowl_v, bowl_f, face_mat, face_group, pan_offset = extract_bowl_surface(
        "data/Tenor Pan only.obj")
    R_level = compute_leveling_rotation("data/Tenor Pan only.obj")
    bowl_v = (R_level @ bowl_v.T).T

    is_dw = classify_drum_wall(bowl_v, bowl_f)
    dw_vi = set()
    for fi, face in enumerate(bowl_f):
        if is_dw[fi]:
            for vi in face:
                dw_vi.add(vi)
    dw_v = bowl_v[sorted(dw_vi)]
    drum_r = float(np.sqrt(dw_v[:, 0]**2 + dw_v[:, 2]**2).max())
    y_min_full = float(bowl_v[:, 1].min())
    rim_y = float(dw_v[:, 1].max())
    y_base_top = y_min_full + BASE_THICKNESS
    print(f"  Drum R={drum_r:.1f}mm, Y=[{y_min_full:.1f}, {rim_y:.1f}]mm")
    print(f"  Base: [{y_min_full:.1f}, {y_base_top:.1f}]mm")
    print(f"  Shell starts at Y={y_base_top:.1f}mm")

    # --------------------------------------------------------
    # Phase 2: Subdivide playing surface for heightmap
    # --------------------------------------------------------
    print("\nPhase 2: Subdivide playing surface...")
    ps_faces = [bowl_f[fi] for fi in range(len(bowl_f)) if not is_dw[fi]]
    ps_groups = [face_group[fi] for fi in range(len(bowl_f)) if not is_dw[fi]]
    note_pan_groups = set(p for (g, p) in NOTE_MAPPING.keys())
    # Inner ring pads include groove geometry — pocket the groove area too
    for (g, p), info in NOTE_MAPPING.items():
        if info[2] == 'inner':
            note_pan_groups.add(g)  # add grove group for inner ring
    ps_is_pocket = [g in note_pan_groups for g in ps_groups]
    ps_v, ps_f = reindex_mesh(bowl_v, ps_faces)
    for rnd in range(SUBDIVIDE_ROUNDS):
        ps_v, ps_f, ps_is_pocket = subdivide_mesh(ps_v, ps_f, ps_is_pocket)
    print(f"  {len(ps_v)}v {len(ps_f)}f, {sum(ps_is_pocket)} pocket faces")

    # Mark pocket vertices (vertex where ALL adjacent faces are pocket faces)
    n_ps = len(ps_v)
    v_pocket_count = np.zeros(n_ps, dtype=int)
    v_face_count = np.zeros(n_ps, dtype=int)
    for fi, face in enumerate(ps_f):
        for vi in face:
            v_face_count[vi] += 1
            if ps_is_pocket[fi]:
                v_pocket_count[vi] += 1
    v_is_pocket = (v_pocket_count == v_face_count) & (v_face_count > 0)

    # --------------------------------------------------------
    # Phase 3: Load notepad / hardware data
    # --------------------------------------------------------
    print("\nPhase 3: Notepad data...")
    with open("data/pan_centroid_offset.json") as f:
        pan_centroid_offset = np.array(json.load(f)["centroid_offset_mm"])

    props = json.load(open('data/notepads/notepad_properties.json'))

    pads = []
    screw_holes = []
    for p in props:
        idx = p['index']
        ring = p['ring']

        # Transform centroid + normal to body coords
        centroid_body = R_level @ (np.array(p['centroid']) - pan_centroid_offset)
        normal_body = R_level @ np.array(p['normal'])
        normal_body /= np.linalg.norm(normal_body)

        # Surface centroid (centroid is at PAN_THICKNESS/2 below surface)
        surface_centroid = centroid_body + normal_body * (PAN_THICKNESS / 2)

        # Hardware clearance radius (largest mechanism that passes through)
        if ring == 'outer':
            # Mount base (no sleeve) — larger than the boss cylinder
            boss_r = MOUNT_BASE_OUTER_R + BOSS_HOLE_TOLERANCE
        else:
            # Boss cylinder (larger than push cap at 7.35mm radius)
            boss_r = (CENTRAL_MOUNT_INNER_DIAMETER / 2 + CENTRAL_MOUNT_WALL_THICKNESS
                      + CENTRAL_MOUNT_THREAD_DEPTH + BOSS_HOLE_TOLERANCE)

        # Screw hole positions in body coords
        for hp in p.get('hole_positions', []):
            hp_body = R_level @ (np.array(hp) - pan_centroid_offset)
            screw_holes.append(hp_body)

        pads.append({
            'idx': idx,
            'ring': ring,
            'surface_centroid': surface_centroid,
            'normal': normal_body,
            'boss_r': boss_r,
        })
    print(f"  {len(pads)} pads, {len(screw_holes)} screw holes")

    # --------------------------------------------------------
    # Phase 4: Build heightmaps
    # --------------------------------------------------------
    print("\nPhase 4: Heightmaps...")
    margin = res * 2
    x_range = np.arange(-drum_r - margin, drum_r + margin + res, res)
    z_range = np.arange(-drum_r - margin, drum_r + margin + res, res)
    y_range = np.arange(y_min_full - margin, rim_y + margin + res, res)
    nx, ny, nz = len(x_range), len(y_range), len(z_range)
    print(f"  Grid: {nx}x{ny}x{nz} = {nx * ny * nz / 1e6:.0f}M voxels ({res}mm)")

    XI, ZI = np.meshgrid(x_range, z_range, indexing='ij')

    # Shell top: outer surface for non-pad areas, pocket floor for pad areas
    shell_top_y = ps_v[:, 1].copy()
    for i in range(n_ps):
        if v_is_pocket[i]:
            shell_top_y[i] -= POCKET_DEPTH

    # Shell bottom: outer surface minus full shell thickness
    shell_bot_y = ps_v[:, 1] - SHELL_THICKNESS

    # Interpolate on a coarser grid (1mm), then upscale with bilinear zoom.
    # The surface is smooth — 1mm captures the shape; 0.2mm is for voxel detail.
    INTERP_RES = max(res, 1.0)  # never coarser than 1mm
    if INTERP_RES > res:
        cx_range = np.arange(x_range[0], x_range[-1] + INTERP_RES, INTERP_RES)
        cz_range = np.arange(z_range[0], z_range[-1] + INTERP_RES, INTERP_RES)
        CXI, CZI = np.meshgrid(cx_range, cz_range, indexing='ij')
        cnx, cnz = len(cx_range), len(cz_range)
        print(f"  Coarse grid: {cnx}x{cnz} ({INTERP_RES}mm) → zoom to {nx}x{nz}")
    else:
        CXI, CZI = XI, ZI
        cnx, cnz = nx, nz

    print("  Building Delaunay triangulation (1.1M points)...")
    tri = Delaunay(ps_v[:, [0, 2]])
    print("  Interpolating shell top...")
    interp_top = LinearNDInterpolator(tri, shell_top_y)
    top_coarse = interp_top(CXI, CZI)
    del interp_top
    print("  Interpolating shell bottom...")
    interp_bot = LinearNDInterpolator(tri, shell_bot_y)
    bot_coarse = interp_bot(CXI, CZI)
    del interp_bot, tri

    if INTERP_RES > res:
        del CXI, CZI
        print("  Upscaling heightmaps to fine grid...")
        zf0 = nx / cnx
        zf1 = nz / cnz
        top_height = ndimage_zoom(np.nan_to_num(top_coarse, nan=rim_y), (zf0, zf1), order=1)
        bot_height = ndimage_zoom(np.nan_to_num(bot_coarse, nan=rim_y - SHELL_THICKNESS), (zf0, zf1), order=1)
        # Trim/pad to exact grid size
        top_height = top_height[:nx, :nz]
        bot_height = bot_height[:nx, :nz]
        del top_coarse, bot_coarse
    else:
        top_height = top_coarse
        bot_height = bot_coarse

    RR = np.sqrt(XI ** 2 + ZI ** 2)

    # Build per-angle outer radius map from actual drum wall vertices
    # so the wall follows the source model profile exactly
    dw_theta = np.arctan2(dw_v[:, 2], dw_v[:, 0])
    dw_r = np.sqrt(dw_v[:, 0] ** 2 + dw_v[:, 2] ** 2)
    N_ANGLE_BINS = 720
    angle_edges = np.linspace(-np.pi, np.pi, N_ANGLE_BINS + 1)
    max_r_by_angle = np.full(N_ANGLE_BINS, drum_r)
    for ai in range(N_ANGLE_BINS):
        mask = (dw_theta >= angle_edges[ai]) & (dw_theta < angle_edges[ai + 1])
        if mask.any():
            max_r_by_angle[ai] = dw_r[mask].max()

    grid_theta = np.arctan2(ZI, XI)
    angle_idx = np.clip(
        ((grid_theta + np.pi) / (2 * np.pi) * N_ANGLE_BINS).astype(int),
        0, N_ANGLE_BINS - 1)
    wall_outer_r = max_r_by_angle[angle_idx]  # actual outer R per grid cell
    inside_drum = RR <= wall_outer_r  # use actual profile, not uniform drum_r

    # Fill NaN inside drum with rim values
    top_height[np.isnan(top_height) & inside_drum] = rim_y
    bot_height[np.isnan(bot_height) & inside_drum] = rim_y - SHELL_THICKNESS

    del XI, ZI, shell_top_y, shell_bot_y
    gc.collect()

    # --------------------------------------------------------
    # Phase 5: Fill shell voxels (vectorized)
    # --------------------------------------------------------
    print("\nPhase 5: Fill shell...")
    grid = np.zeros((nx, ny, nz), dtype=np.uint8)

    # Clamp bot_height to above base plate
    bot_clamped = np.maximum(bot_height, y_base_top)

    # Convert heightmaps to Y-index arrays
    iy_top_map = np.clip(((top_height - y_range[0]) / res).astype(np.int32), 0, ny - 1)
    iy_bot_map = np.clip(((bot_clamped - y_range[0]) / res).astype(np.int32), 0, ny - 1)

    # Valid cells: inside drum, no NaN
    valid = inside_drum & ~np.isnan(top_height) & ~np.isnan(bot_height)

    # Fill playing surface shell — chunked numpy (no Python per-column loop)
    CHUNK = 64  # X-slices per chunk
    y_idx = np.arange(ny, dtype=np.int32)  # [ny]
    n_chunks = (nx + CHUNK - 1) // CHUNK
    print(f"  Filling shell ({n_chunks} chunks of {CHUNK})...")
    for ci in range(n_chunks):
        ix0 = ci * CHUNK
        ix1 = min(ix0 + CHUNK, nx)
        # Broadcast: y_idx[1,ny,1] vs maps[chunk,1,nz]
        chunk_valid = valid[ix0:ix1, np.newaxis, :]         # [chunk,1,nz]
        chunk_bot = iy_bot_map[ix0:ix1, np.newaxis, :]      # [chunk,1,nz]
        chunk_top = iy_top_map[ix0:ix1, np.newaxis, :]      # [chunk,1,nz]
        y_broad = y_idx[np.newaxis, :, np.newaxis]           # [1,ny,1]
        fill = chunk_valid & (y_broad >= chunk_bot) & (y_broad <= chunk_top)
        grid[ix0:ix1] = fill.astype(np.uint8)
        del fill, chunk_valid, chunk_bot, chunk_top

    # Drum wall shell — follow actual wall profile (chunked)
    iy_base_top_idx = max(0, int((y_base_top - y_range[0]) / res))
    iy_rim = min(ny - 1, int((rim_y - y_range[0]) / res))

    wall_mask = (RR >= wall_outer_r - SHELL_THICKNESS) & (RR <= wall_outer_r)
    for ci in range(n_chunks):
        ix0 = ci * CHUNK
        ix1 = min(ix0 + CHUNK, nx)
        wm = wall_mask[ix0:ix1, :, np.newaxis] if wall_mask.ndim == 2 else wall_mask[ix0:ix1]
        # wall_mask is 2D [nx,nz], expand and fill Y range
        for local_ix in range(ix1 - ix0):
            wz = np.where(wall_mask[ix0 + local_ix])[0]
            if len(wz) > 0:
                grid[ix0 + local_ix, iy_base_top_idx:iy_rim + 1, wz] = 1

    n_shell = int(grid.sum())
    print(f"  {n_shell / 1e6:.1f}M shell voxels")

    del bot_clamped, iy_top_map, iy_bot_map, valid
    del wall_mask
    gc.collect()

    # --------------------------------------------------------
    # Phase 6: Cut boss holes through shell (vectorized)
    # --------------------------------------------------------
    print("\nPhase 6: Boss holes...")
    for pad in pads:
        cx = pad['surface_centroid'][0]
        cz = pad['surface_centroid'][2]
        br = pad['boss_r']

        ix_min = max(0, int((cx - br - x_range[0]) / res) - 1)
        ix_max = min(nx - 1, int((cx + br - x_range[0]) / res) + 2)
        iz_min = max(0, int((cz - br - z_range[0]) / res) - 1)
        iz_max = min(nz - 1, int((cz + br - z_range[0]) / res) + 2)

        # Vectorized distance check
        local_x = x_range[ix_min:ix_max + 1] - cx
        local_z = z_range[iz_min:iz_max + 1] - cz
        DX, DZ = np.meshgrid(local_x, local_z, indexing='ij')
        mask = (DX ** 2 + DZ ** 2) <= br ** 2
        for di in range(mask.shape[0]):
            for dj in range(mask.shape[1]):
                if mask[di, dj]:
                    grid[ix_min + di, :, iz_min + dj] = 0

    n_after_boss = int(grid.sum())
    print(f"  Cleared {(n_shell - n_after_boss) / 1e6:.1f}M voxels for {len(pads)} boss holes")

    # --------------------------------------------------------
    # Phase 7: Cut pad screw holes (M2 through shell)
    # --------------------------------------------------------
    print("\nPhase 7: Pad screw holes...")
    for h in screw_holes:
        hx, hz = h[0], h[2]
        ix_min = max(0, int((hx - SURFACE_SCREW_R - x_range[0]) / res) - 1)
        ix_max = min(nx - 1, int((hx + SURFACE_SCREW_R - x_range[0]) / res) + 2)
        iz_min = max(0, int((hz - SURFACE_SCREW_R - z_range[0]) / res) - 1)
        iz_max = min(nz - 1, int((hz + SURFACE_SCREW_R - z_range[0]) / res) + 2)

        local_x = x_range[ix_min:ix_max + 1] - hx
        local_z = z_range[iz_min:iz_max + 1] - hz
        DX, DZ = np.meshgrid(local_x, local_z, indexing='ij')
        mask = (DX ** 2 + DZ ** 2) <= SURFACE_SCREW_R ** 2
        for di in range(mask.shape[0]):
            for dj in range(mask.shape[1]):
                if mask[di, dj]:
                    grid[ix_min + di, :, iz_min + dj] = 0

    n_after_screws = int(grid.sum())
    print(f"  {len(screw_holes)} M2 through-holes ({SURFACE_SCREW_R * 2:.1f}mm dia)")
    print(f"  Removed {(n_after_boss - n_after_screws) / 1e6:.2f}M voxels")

    # --------------------------------------------------------
    # Phase 8: Create attachment bosses inside drum wall
    # --------------------------------------------------------
    print("\nPhase 8: Attachment bosses...")
    base_screw_r = drum_r - BASE_SCREW_INSET
    attach_positions = []
    iy_att_bot = max(0, int((y_base_top - y_range[0]) / res))
    iy_att_top = min(ny - 1, int((y_base_top + ATTACH_BOSS_HEIGHT - y_range[0]) / res))
    iy_pilot_top = min(ny - 1, int((y_base_top + M3_PILOT_DEPTH - y_range[0]) / res))

    for si in range(BASE_SCREW_N):
        angle = 2 * math.pi * si / BASE_SCREW_N
        sx = base_screw_r * math.cos(angle)
        sz = base_screw_r * math.sin(angle)
        attach_positions.append((sx, sz))

        # Fill boss cylinder
        ix_min = max(0, int((sx - ATTACH_BOSS_R - x_range[0]) / res) - 1)
        ix_max = min(nx - 1, int((sx + ATTACH_BOSS_R - x_range[0]) / res) + 2)
        iz_min = max(0, int((sz - ATTACH_BOSS_R - z_range[0]) / res) - 1)
        iz_max = min(nz - 1, int((sz + ATTACH_BOSS_R - z_range[0]) / res) + 2)

        local_x = x_range[ix_min:ix_max + 1] - sx
        local_z = z_range[iz_min:iz_max + 1] - sz
        DX, DZ = np.meshgrid(local_x, local_z, indexing='ij')
        boss_mask = (DX ** 2 + DZ ** 2) <= ATTACH_BOSS_R ** 2
        pilot_mask = (DX ** 2 + DZ ** 2) <= M3_PILOT_R ** 2

        for di in range(boss_mask.shape[0]):
            for dj in range(boss_mask.shape[1]):
                if boss_mask[di, dj]:
                    grid[ix_min + di, iy_att_bot:iy_att_top + 1, iz_min + dj] = 1
                if pilot_mask[di, dj]:
                    grid[ix_min + di, iy_att_bot:iy_pilot_top + 1, iz_min + dj] = 0

    print(f"  {BASE_SCREW_N} bosses at R={base_screw_r:.0f}mm")
    print(f"  Boss: {ATTACH_BOSS_R * 2:.0f}mm dia, {ATTACH_BOSS_HEIGHT:.0f}mm tall")
    print(f"  M3 pilot: {M3_PILOT_R * 2:.1f}mm dia, {M3_PILOT_DEPTH:.0f}mm deep")

    n_final_shell = int(grid.sum())
    print(f"  Shell total: {n_final_shell / 1e6:.1f}M voxels")

    # --------------------------------------------------------
    # Phase 9: Base plate (separate voxel grid, vectorized)
    # --------------------------------------------------------
    print("\nPhase 9: Base plate...")
    iy_base_bot_idx = max(0, int((y_min_full - y_range[0]) / res))
    iy_base_top_fill = min(ny - 1, int((y_base_top - y_range[0]) / res))

    base_grid = np.zeros((nx, ny, nz), dtype=np.uint8)
    # Vectorized base fill: set Y range for all inside-drum columns
    for ix in range(nx):
        cols = np.where(inside_drum[ix])[0]
        if len(cols) > 0:
            base_grid[ix, iy_base_bot_idx:iy_base_top_fill + 1, cols] = 1

    # Drill M3 clearance + countersink through base
    for sx, sz in attach_positions:
        ix_min = max(0, int((sx - M3_CSINK_R - x_range[0]) / res) - 1)
        ix_max = min(nx - 1, int((sx + M3_CSINK_R - x_range[0]) / res) + 2)
        iz_min = max(0, int((sz - M3_CSINK_R - z_range[0]) / res) - 1)
        iz_max = min(nz - 1, int((sz + M3_CSINK_R - z_range[0]) / res) + 2)

        local_x = x_range[ix_min:ix_max + 1] - sx
        local_z = z_range[iz_min:iz_max + 1] - sz
        DX, DZ = np.meshgrid(local_x, local_z, indexing='ij')
        R_local = np.sqrt(DX ** 2 + DZ ** 2)
        clear_mask = R_local <= M3_CLEAR_R

        for di in range(R_local.shape[0]):
            for dj in range(R_local.shape[1]):
                r_at = R_local[di, dj]
                if clear_mask[di, dj]:
                    # Clearance through entire base
                    base_grid[ix_min + di, iy_base_bot_idx:iy_base_top_fill + 1, iz_min + dj] = 0
                elif r_at <= M3_CSINK_R:
                    # Conical countersink from bottom
                    for iy in range(iy_base_bot_idx, iy_base_top_fill + 1):
                        depth = (iy - iy_base_bot_idx) * res
                        if depth > M3_CSINK_DEPTH:
                            break
                        r_allowed = M3_CSINK_R - (M3_CSINK_R - M3_CLEAR_R) * (depth / M3_CSINK_DEPTH)
                        if r_at <= r_allowed:
                            base_grid[ix_min + di, iy, iz_min + dj] = 0

    n_base = int(base_grid.sum())
    print(f"  Base: {n_base / 1e6:.1f}M voxels")
    print(f"  {len(attach_positions)} M3x20 countersunk holes")

    del top_height, bot_height, inside_drum, RR
    gc.collect()

    # --------------------------------------------------------
    # Phase 10: Marching cubes → mesh
    # --------------------------------------------------------
    print("\nPhase 10: Marching cubes...")

    # Drum shell
    padded = np.pad(grid, 1, mode='constant', constant_values=0)
    del grid
    gc.collect()
    verts_shell, faces_shell, _, _ = marching_cubes(
        padded, level=0.5, spacing=(res, res, res))
    del padded
    gc.collect()
    verts_shell[:, 0] += x_range[0] - res
    verts_shell[:, 1] += y_range[0] - res
    verts_shell[:, 2] += z_range[0] - res
    print(f"  Shell: {len(verts_shell):,}v, {len(faces_shell):,}f")

    # Base plate
    padded_base = np.pad(base_grid, 1, mode='constant', constant_values=0)
    del base_grid
    gc.collect()
    verts_base, faces_base, _, _ = marching_cubes(
        padded_base, level=0.5, spacing=(res, res, res))
    del padded_base
    gc.collect()
    verts_base[:, 0] += x_range[0] - res
    verts_base[:, 1] += y_range[0] - res
    verts_base[:, 2] += z_range[0] - res
    print(f"  Base:  {len(verts_base):,}v, {len(faces_base):,}f")

    # --------------------------------------------------------
    # Phase 11: Export
    # --------------------------------------------------------
    print("\nPhase 11: Export...")

    # OBJ with named objects
    obj_path = 'data/pan_surface_up_thick.obj'
    with open(obj_path, 'w') as f:
        f.write("# Pan surface thickened shell + base plate\n")
        f.write(f"# Shell: {SHELL_THICKNESS}mm thick, pockets: {POCKET_DEPTH}mm deep\n")
        f.write(f"# Base: {BASE_THICKNESS}mm, {BASE_SCREW_N}x M3x20 countersunk\n")
        f.write("# Units: mm\n\n")

        f.write("o DrumShell\n")
        for v in verts_shell:
            f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        f.write("\n")
        for face in faces_shell:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

        n_shell_v = len(verts_shell)
        f.write("\no BasePlate\n")
        for v in verts_base:
            f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        f.write("\n")
        for face in faces_base:
            f.write(f"f {face[0] + 1 + n_shell_v} {face[1] + 1 + n_shell_v} "
                    f"{face[2] + 1 + n_shell_v}\n")

    print(f"  Saved: {obj_path}")

    # STL (combined)
    stl_path = 'data/pan_surface_up_thick.stl'
    all_verts = np.vstack([verts_shell, verts_base])
    all_faces = (list(faces_shell) +
                 [[f[0] + n_shell_v, f[1] + n_shell_v, f[2] + n_shell_v]
                  for f in faces_base])
    write_stl(stl_path, all_verts, all_faces)
    print(f"  Saved: {stl_path}")

    print(f"\n{'=' * 60}")
    print(f"Done!")
    print(f"  Shell: {len(verts_shell):,} verts, {len(faces_shell):,} faces")
    print(f"  Base:  {len(verts_base):,} verts, {len(faces_base):,} faces")
    print(f"  Pads:  {len(pads)} boss holes, {len(screw_holes)} screw holes")
    print(f"  Base:  {BASE_SCREW_N} M3x20 attachment screws")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
