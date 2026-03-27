#!/usr/bin/env python3
"""
Generate solid pan body via voxelisation.

Features:
- Bowl surface with 5mm pad pockets (flush fit)
- Hardware through-holes (MountBase+Sleeve, buffered 2.5mm, clipped to pads)
- Screw pilot holes (1.8mm, 4mm deep)
- Cylindrical base plate (10mm thick, separate piece)
- Wiring void (cylinder inset 15mm from edge, 50mm tall above base)

Coordinates match pan_surface_up.obj and assembly_view.obj (R_level, Y=drum axis).

Usage:
    python generate_pan_body.py [--res=0.3] [--sdf] [--sigma=0.5] [--save-grid]
                                [--grid-only]

    --sdf          Use signed distance field for smooth mesh extraction
                   (eliminates staircase artifacts while preserving sharp edges)
    --sigma=X      Gaussian smoothing sigma in voxels (implies --sdf, default 0.5)
    --save-grid    Save binary grids to .npz for fast re-extraction
    --grid-only    Save grid and skip mesh extraction (use extract_pieces.py later)
"""

import numpy as np
import json
import struct
import gc
import sys
import math
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from matplotlib.path import Path as MplPath
from skimage.measure import marching_cubes

from generate_sector import extract_bowl_surface
from generate_notepad import (
    compute_leveling_rotation, NOTE_BY_INDEX, NOTE_MAPPING, PAN_THICKNESS
)
from generate_quarter import classify_drum_wall, reindex_mesh, subdivide_mesh
from generate_cylinders import (
    CYL_OUTER_R, CYL_INNER_R, CYL_SPLIT_R,
    N_STRUTS, STRUT_WALL, STRUT_HALF_W,
    STRUT_ANGLES_DEG, STRUT_ANGLES_RAD,
    KEY_WIDTH, KEY_DEPTH,
    WIRE_HOLE_R, WIRE_HOLE_CENTER_HEIGHT,
    STRUT_WIRE_R_POS, WIRE_CYL_ANGLES_RAD,
    SCREW_PILOT_R, SCREW_CLEAR_R, SCREW_DEPTH, SCREW_EDGE_MAX,
    CYL_SCREW_R_INNER, CYL_SCREW_R_OUTER, CYL_SCREW_N,
    STRUT_SCREW_SPACING, STRUT_SCREW_R_START,
)

# ============================================================
# Parameters
# ============================================================

POCKET_DEPTH = PAN_THICKNESS   # 5mm — pad sits flush
POCKET_TOLERANCE = 1.0         # mm extra around pad footprint for fit
HW_BUFFER = 2.5                # mm clearance around hardware
PILOT_R = 1.0                  # mm — M2 pilot hole radius (2mm dia)
PILOT_DEPTH = 5.0              # mm — visible from pocket floor
# M2 countersink + plug bore at pocket floor (DIN 965 flat-head)
CSINK_TOP_R = 2.0              # mm — head radius (4.0mm dia)
CSINK_TAPER_DEPTH = 1.0        # mm — 90° cone depth
CSINK_PLUG_DEPTH = 2.0         # mm — cylindrical bore above taper for plug
BASE_THICKNESS = 10.0          # mm — base plate
VOID_INSET = 15.0              # mm from drum edge
VOID_CLEARANCE = 20.0          # mm below underside of pan surface
SUBDIVIDE_ROUNDS = 2

# Groove embedding — pad top sits this far below playing surface
# to match the groove inner edge height
GROOVE_STEP = 0.1                  # mm — must match STEP_INNER in generate_grooves.py

# Base attachment screws
BASE_SCREW_N = 8               # number of M3 screws around perimeter
BASE_SCREW_INSET = 12.0        # mm from drum outer edge
BASE_SCREW_PILOT_R = 1.25      # mm — M3 tapped pilot hole radius
BASE_SCREW_CLEAR_R = 1.7       # mm — M3 clearance in base plate
BASE_SCREW_DEPTH = 8.0         # mm — pilot hole depth into drum body


def _sdf_extract(padded, res, sigma):
    """SDF-based mesh extraction from padded binary grid.

    Computes signed distance field via Euclidean distance transform, then
    extracts the zero-level isosurface with marching cubes.  For grids > 2B
    voxels the float64→float32 conversion is spilled to disk so peak RAM
    stays below ~57 GB.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    shape = padded.shape
    pb = padded.view(np.bool_)

    if padded.size < 2_000_000_000:
        # --- in-memory path ---
        print("  SDF interior...")
        sdf = distance_transform_edt(pb).astype(np.float32)
        print("  SDF exterior...")
        np.logical_not(pb, out=pb)
        sdf -= distance_transform_edt(pb).astype(np.float32)
    else:
        # --- disk-spill path ---
        import os, tempfile
        td = tempfile.mkdtemp(prefix='pan_sdf_')
        pin = os.path.join(td, 'in.f32')
        pex = os.path.join(td, 'ex.f32')

        print(f"  SDF interior (disk-spill, {padded.size / 1e9:.1f}B voxels)...")
        d = distance_transform_edt(pb)
        with open(pin, 'wb') as f:
            for i in range(shape[0]):
                f.write(d[i].astype(np.float32).tobytes())
        del d; gc.collect()

        print("  SDF exterior (disk-spill)...")
        np.logical_not(pb, out=pb)
        d = distance_transform_edt(pb)
        with open(pex, 'wb') as f:
            for i in range(shape[0]):
                f.write(d[i].astype(np.float32).tobytes())
        del d; gc.collect()

        print("  Combining SDF from disk...")
        mi = np.memmap(pin, dtype=np.float32, mode='r', shape=shape)
        me = np.memmap(pex, dtype=np.float32, mode='r', shape=shape)
        sdf = np.empty(shape, dtype=np.float32)
        cs = max(1, shape[0] // 20)
        for i in range(0, shape[0], cs):
            e = min(i + cs, shape[0])
            sdf[i:e] = mi[i:e] - me[i:e]
        del mi, me; gc.collect()
        os.remove(pin); os.remove(pex); os.rmdir(td)

    if sigma > 0:
        print(f"  Gaussian sigma={sigma:.1f}...")
        gaussian_filter(sdf, sigma=sigma, output=sdf)

    print("  Marching cubes (SDF level=0)...")
    verts, faces, _, _ = marching_cubes(sdf, level=0.0, spacing=(res, res, res))
    del sdf; gc.collect()
    return verts, faces


def main():
    # Parse args
    res = 0.5
    use_sdf = '--sdf' in sys.argv
    sigma = 0.0
    save_grid = '--save-grid' in sys.argv
    grid_only = '--grid-only' in sys.argv  # save grid and skip extraction
    for arg in sys.argv[1:]:
        if arg.startswith('--res='):
            res = float(arg.split('=')[1])
        elif arg.startswith('--sigma='):
            sigma = float(arg.split('=')[1])
            use_sdf = True
    if use_sdf and sigma == 0:
        sigma = 0.5
    if use_sdf:
        from scipy.ndimage import distance_transform_edt, gaussian_filter

    print("=" * 60)
    print(f"Pan Body — {res}mm, {HW_BUFFER}mm buffer, {POCKET_DEPTH}mm pockets")
    print(f"  Base: {BASE_THICKNESS}mm, Void: {VOID_INSET}mm inset, {VOID_CLEARANCE}mm below surface")
    if use_sdf:
        print(f"  SDF extraction, sigma={sigma:.1f} voxels ({sigma*res:.2f}mm)")
    print("=" * 60)

    # Phase 1: Load + level
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
    y_min = float(bowl_v[:, 1].min())
    rim_y = float(dw_v[:, 1].max())
    print(f"  R={drum_r:.0f}, Y=[{y_min:.0f},{rim_y:.0f}]")

    # Phase 2: Subdivide playing surface
    print("\nPhase 2: Subdivide...")
    ps_faces = [bowl_f[fi] for fi in range(len(bowl_f)) if not is_dw[fi]]
    ps_groups = [face_group[fi] for fi in range(len(bowl_f)) if not is_dw[fi]]
    note_pan_groups = set(p for (g, p) in NOTE_MAPPING.keys())
    ps_is_pocket = [g in note_pan_groups for g in ps_groups]
    ps_v, ps_f = reindex_mesh(bowl_v, ps_faces)
    for rnd in range(SUBDIVIDE_ROUNDS):
        ps_v, ps_f, ps_is_pocket = subdivide_mesh(ps_v, ps_f, ps_is_pocket)
    print(f"  {len(ps_v)}v {len(ps_f)}f, {sum(ps_is_pocket)} pocket faces")

    # Phase 3: Assembly hardware
    print("\nPhase 3: Hardware...")
    a_verts = []
    a_objects = {}
    a_current = None
    with open('data/notepads/assembly_view.obj') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                p = line.split()
                a_verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith('o '):
                a_current = line[2:]
                a_objects[a_current] = set()
            elif line.startswith('f ') and a_current:
                for p in line.split()[1:]:
                    try:
                        a_objects[a_current].add(int(p.split('/')[0]) - 1)
                    except ValueError:
                        pass
    a_verts = np.array(a_verts)

    hw_shapes = {}
    for note in sorted(NOTE_BY_INDEX.keys()):
        hw_xz = []
        for prefix in ['MountBase_', 'Sleeve_']:
            key = prefix + note
            if key in a_objects:
                hv = a_verts[sorted(a_objects[key])]
                hw_xz.append(hv[:, [0, 2]])
        if not hw_xz:
            continue
        all_hw = np.vstack(hw_xz)
        hull = ConvexHull(all_hw)
        hw_poly = Polygon(all_hw[hull.vertices])

        # Make mask symmetric about pad's long axis (PCA) for PCB clearance
        pd_key = f'Pad_{note}'
        if pd_key in a_objects:
            pv = a_verts[sorted(a_objects[pd_key])]
            pad_xz = pv[:, [0, 2]]
            pad_centroid_xz = pad_xz.mean(axis=0)
            centered_xz = pad_xz - pad_centroid_xz
            cov_xz = centered_xz.T @ centered_xz
            eigvals_xz, eigvecs_xz = np.linalg.eigh(cov_xz)
            long_axis = eigvecs_xz[:, np.argmax(eigvals_xz)]

            # Reflect hw_poly about line through pad centroid along long_axis
            coords = np.array(hw_poly.exterior.coords)
            rel = coords - pad_centroid_xz
            proj = np.outer(rel @ long_axis, long_axis)
            reflected = pad_centroid_xz + 2 * proj - rel
            mirror_poly = Polygon(reflected)
            hw_poly = hw_poly.union(mirror_poly)
            if hw_poly.geom_type == 'MultiPolygon':
                hw_poly = hw_poly.convex_hull

            pad_poly = Polygon(pad_xz[ConvexHull(pad_xz).vertices])
            clipped = pad_poly.intersection(hw_poly)
        else:
            clipped = hw_poly
        if not clipped.is_empty:
            buffered = clipped.buffer(HW_BUFFER)
            if pd_key in a_objects:
                buffered = pad_poly.intersection(buffered)
            if buffered.geom_type == 'Polygon' and not buffered.is_empty:
                hw_shapes[note] = MplPath(np.array(buffered.exterior.coords))
    print(f"  {len(hw_shapes)} hw holes (symmetric, buffered {HW_BUFFER}mm)")

    props = json.load(open('data/notepads/notepad_properties.json'))
    offset = np.array(
        json.load(open('data/pan_centroid_offset.json'))['centroid_offset_mm'])
    screw_holes = []
    for p in props:
        for hp in p.get('hole_positions', []):
            h = R_level @ (np.array(hp) - offset)
            screw_holes.append((h[0], h[1], h[2]))
    print(f"  {len(screw_holes)} screw holes")

    # Compute rim clip radius: inner R of outer pad top surfaces.
    # The voxel body stops here; the rim mesh covers R > rim_clip_r.
    rim_clip_r = float('inf')
    for name, vis in a_objects.items():
        if name.startswith('Pad_O'):
            pv = a_verts[sorted(vis)]
            y_max = float(pv[:, 1].max())
            top = pv[pv[:, 1] > y_max - 5.0]
            r_min = float(np.sqrt(top[:, 0]**2 + top[:, 2]**2).min())
            rim_clip_r = min(rim_clip_r, r_min)
    print(f"  Rim clip R = {rim_clip_r:.1f}mm (outer pad top inner edge)")

    # Phase 3b: Load groove top surfaces for embedding into body
    print("\nPhase 3b: Groove surfaces...")
    from pathlib import Path as PathLib
    groove_top_surfaces = {}
    for ring_name in ['outer', 'central', 'inner']:
        groove_path = f'data/grooves/grooves_{ring_name}.obj'
        if not PathLib(groove_path).exists():
            continue
        all_gv = []
        current_gobj = None
        gobj_ranges = {}
        with open(groove_path) as gf:
            for line in gf:
                line = line.strip()
                if line.startswith('o '):
                    if current_gobj is not None:
                        gobj_ranges[current_gobj] = (gobj_ranges[current_gobj][0], len(all_gv))
                    current_gobj = line[2:]
                    gobj_ranges[current_gobj] = (len(all_gv), len(all_gv))
                elif line.startswith('v '):
                    p = line.split()
                    all_gv.append([float(p[1]), float(p[2]), float(p[3])])
        if current_gobj is not None:
            gobj_ranges[current_gobj] = (gobj_ranges[current_gobj][0], len(all_gv))
        if not all_gv:
            continue
        all_gv = np.array(all_gv)
        for gobj_name, (v_start, v_end) in gobj_ranges.items():
            note_key = gobj_name.replace('Groove_', '')
            n_total = v_end - v_start
            n_top = n_total // 2  # first half is top surface (tapered)
            groove_top_surfaces[note_key] = all_gv[v_start:v_start + n_top]
    print(f"  {len(groove_top_surfaces)} groove surfaces loaded")

    # Phase 4: Heightmap + pockets
    print("\nPhase 4: Heightmap...")
    n_ps = len(ps_v)
    v_pocket_count = np.zeros(n_ps, dtype=int)
    v_face_count = np.zeros(n_ps, dtype=int)
    for fi, face in enumerate(ps_f):
        for vi in face:
            v_face_count[vi] += 1
            if ps_is_pocket[fi]:
                v_pocket_count[vi] += 1
    v_is_pocket = (v_pocket_count == v_face_count) & (v_face_count > 0)

    surface_y = ps_v[:, 1].copy()
    for i in range(n_ps):
        if v_is_pocket[i]:
            surface_y[i] -= POCKET_DEPTH

    margin = res * 2
    x_range = np.arange(-drum_r - margin, drum_r + margin + res, res)
    z_range = np.arange(-drum_r - margin, drum_r + margin + res, res)
    y_range = np.arange(y_min - margin, rim_y + margin + res, res)
    nx, ny, nz = len(x_range), len(y_range), len(z_range)
    print(f"  Grid: {nx}x{ny}x{nz} = {nx*ny*nz/1e6:.0f}M ({res}mm)")

    XI, ZI = np.meshgrid(x_range, z_range, indexing='ij')
    print("  Interpolating...")
    height_map = griddata(ps_v[:, [0, 2]], surface_y, (XI, ZI), method='linear')
    RR = np.sqrt(XI**2 + ZI**2)

    # Compute per-angle max R from ALL bowl vertices so the drum wall is
    # thickened inward (fill stops at the actual surface, not at drum_r).
    print("  Computing radial boundary...")
    bowl_r = np.sqrt(bowl_v[:, 0]**2 + bowl_v[:, 2]**2)
    bowl_theta = np.arctan2(bowl_v[:, 2], bowl_v[:, 0])
    n_bins = 360
    angle_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    r_max_per_bin = np.full(n_bins, 0.0)
    for bi in range(n_bins):
        mask = (bowl_theta >= angle_bins[bi]) & (bowl_theta < angle_bins[bi + 1])
        if mask.any():
            r_max_per_bin[bi] = bowl_r[mask].max()
    # Fill any empty bins with neighbors
    for bi in range(n_bins):
        if r_max_per_bin[bi] == 0:
            r_max_per_bin[bi] = r_max_per_bin[bi - 1]
    # Interpolate to grid columns
    bin_centers = 0.5 * (angle_bins[:-1] + angle_bins[1:])
    grid_theta = np.arctan2(ZI, XI)
    surface_r_map = np.interp(grid_theta.ravel(), bin_centers, r_max_per_bin,
                              period=2 * np.pi).reshape(grid_theta.shape)
    inside_drum = RR <= surface_r_map
    print(f"  Surface R: [{surface_r_map.min():.1f}, {surface_r_map.max():.1f}]mm")
    del bowl_r, bowl_theta, surface_r_map, grid_theta

    height_map[np.isnan(height_map) & inside_drum] = rim_y
    col_top = np.full((nx, nz), y_min)
    valid = ~np.isnan(height_map) & inside_drum
    col_top[valid] = height_map[valid]
    iy_top = np.clip(((col_top - y_range[0]) / res).astype(np.int32) + 1, 0, ny)

    # Compute void ceiling: original surface Y - VOID_CLEARANCE per column
    ps_vi_all = sorted(set(vi for fi, f in enumerate(bowl_f) if not is_dw[fi] for vi in f))
    ps_pts = bowl_v[ps_vi_all][:, [0, 2]]
    ps_yvals = bowl_v[ps_vi_all][:, 1]
    XI2, ZI2 = np.meshgrid(x_range, z_range, indexing='ij')
    orig_height = griddata(ps_pts, ps_yvals, (XI2, ZI2), method='nearest')
    void_top_map = orig_height - VOID_CLEARANCE
    del orig_height, XI2, ZI2, ps_pts, ps_yvals, ps_vi_all

    del XI, ZI, height_map, surface_y, ps_v, ps_f, v_pocket_count, v_face_count
    gc.collect()

    # Phase 5: Fill solid
    print("\nPhase 5: Fill...")
    grid = np.zeros((nx, ny, nz), dtype=np.uint8)
    for ix in range(nx):
        for iz in range(nz):
            iy_end = iy_top[ix, iz]
            if iy_end > 0 and inside_drum[ix, iz]:
                grid[ix, :iy_end, iz] = 1
    n_solid = int(grid.sum())
    print(f"  {n_solid / 1e6:.0f}M voxels")

    # Phase 5a: Voxelize rim mesh (thickened inward) into the grid.
    # The rim surface at R > rim_clip_r is loaded from the mesh produced by
    # extract_rim.py, thickened inward by PAN_THICKNESS, and merged into the
    # solid grid.  Pad pockets (5b) and hardware holes (7) carve through it.
    print("\nPhase 5a: Rim voxelization...")
    rim_path = 'data/drum_rim.obj'
    from pathlib import Path as PathLib
    if PathLib(rim_path).exists():
        from generate_quarter import compute_face_normals as _cfn
        rim_verts, rim_faces = [], []
        with open(rim_path) as rf:
            for line in rf:
                if line.startswith('v '):
                    p = line.split()
                    rim_verts.append([float(p[1]), float(p[2]), float(p[3])])
                elif line.startswith('f '):
                    face = [int(tok.split('/')[0]) - 1 for tok in line.split()[1:]]
                    rim_faces.append(face)
        rim_verts = np.array(rim_verts)
        # Triangulate quads
        rim_tris = []
        for face in rim_faces:
            for i in range(1, len(face) - 1):
                rim_tris.append([face[0], face[i], face[i + 1]])
        rim_tris_arr = np.array(rim_tris)
        rim_normals = _cfn(rim_verts, rim_tris)

        # For each triangle, compute bounding box and fill voxels that are
        # within PAN_THICKNESS inward from the surface.
        n_rim_filled = 0
        thickness = PAN_THICKNESS
        for ti in range(len(rim_tris)):
            tri = rim_tris_arr[ti]
            v0, v1, v2 = rim_verts[tri[0]], rim_verts[tri[1]], rim_verts[tri[2]]
            n = rim_normals[ti]
            nl = np.linalg.norm(n)
            if nl < 1e-10:
                continue
            n = n / nl
            # Ensure normal points inward (toward drum center)
            centroid = (v0 + v1 + v2) / 3.0
            r_cent = math.sqrt(centroid[0]**2 + centroid[2]**2)
            radial_dir = np.array([centroid[0], 0, centroid[2]])
            radial_dir_n = np.linalg.norm(radial_dir)
            if radial_dir_n > 0:
                radial_dir /= radial_dir_n
            # If normal points outward (same direction as radial), flip it
            if np.dot(n, radial_dir) > 0:
                n = -n

            # Bounding box of the triangle + thickness sweep
            all_pts = np.array([v0, v1, v2,
                                v0 + n * thickness,
                                v1 + n * thickness,
                                v2 + n * thickness])
            bb_min = all_pts.min(axis=0)
            bb_max = all_pts.max(axis=0)

            ix_lo = max(0, int((bb_min[0] - x_range[0]) / res) - 1)
            ix_hi = min(nx - 1, int((bb_max[0] - x_range[0]) / res) + 2)
            iy_lo = max(0, int((bb_min[1] - y_range[0]) / res) - 1)
            iy_hi = min(ny - 1, int((bb_max[1] - y_range[0]) / res) + 2)
            iz_lo = max(0, int((bb_min[2] - z_range[0]) / res) - 1)
            iz_hi = min(nz - 1, int((bb_max[2] - z_range[0]) / res) + 2)

            # Triangle edges for barycentric test
            e0 = v1 - v0
            e1 = v2 - v0
            d00 = np.dot(e0, e0)
            d01 = np.dot(e0, e1)
            d11 = np.dot(e1, e1)
            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-12:
                continue

            for ix in range(ix_lo, ix_hi + 1):
                px = x_range[ix]
                for iz in range(iz_lo, iz_hi + 1):
                    pz = z_range[iz]
                    for iy in range(iy_lo, iy_hi + 1):
                        if grid[ix, iy, iz] == 1:
                            continue
                        py = y_range[iy]
                        p = np.array([px, py, pz])
                        # Signed distance from triangle plane
                        dp = p - v0
                        dist = np.dot(dp, n)
                        if dist < -res or dist > thickness + res:
                            continue
                        # Project onto triangle plane
                        proj = p - dist * n
                        dp2 = proj - v0
                        d20 = np.dot(dp2, e0)
                        d21 = np.dot(dp2, e1)
                        u = (d11 * d20 - d01 * d21) / denom
                        v = (d00 * d21 - d01 * d20) / denom
                        # Inside triangle with small margin
                        margin = res / max(np.linalg.norm(e0), np.linalg.norm(e1), 1.0)
                        if u >= -margin and v >= -margin and (u + v) <= 1.0 + margin:
                            grid[ix, iy, iz] = 1
                            n_rim_filled += 1

        print(f"  {len(rim_tris)} rim triangles, {n_rim_filled} voxels added "
              f"({thickness}mm inward)")
    else:
        print(f"  {rim_path} not found — run extract_rim.py first")

    # Phase 5b: Carve pad pockets from assembly_view Pad footprints
    # Ensures ALL pads (including inner) get proper indentations
    print("\nPhase 5b: Assembly pad pockets...")
    n_pocketed = 0
    for note in sorted(NOTE_BY_INDEX.keys()):
        pd_key = f'Pad_{note}'
        if pd_key not in a_objects:
            continue
        pv = a_verts[sorted(a_objects[pd_key])]
        pad_xz = pv[:, [0, 2]]
        pad_y_top = float(pv[:, 1].max())
        pocket_floor_y = pad_y_top - POCKET_DEPTH - GROOVE_STEP

        try:
            hull = ConvexHull(pad_xz)
            pad_poly = Polygon(pad_xz[hull.vertices])
            # Add tolerance so pad fits easily
            pad_poly_buffered = pad_poly.buffer(POCKET_TOLERANCE)
            if pad_poly_buffered.geom_type == 'Polygon':
                pad_path = MplPath(np.array(pad_poly_buffered.exterior.coords))
            else:
                pad_path = MplPath(pad_xz[hull.vertices])
        except:
            continue

        bounds = pad_path.get_extents()
        ix_min = max(0, int((bounds.x0 - x_range[0]) / res) - 1)
        ix_max = min(nx - 1, int((bounds.x1 - x_range[0]) / res) + 2)
        iz_min = max(0, int((bounds.y0 - z_range[0]) / res) - 1)
        iz_max = min(nz - 1, int((bounds.y1 - z_range[0]) / res) + 2)
        n_ix, n_iz = ix_max - ix_min + 1, iz_max - iz_min + 1
        if n_ix <= 0 or n_iz <= 0:
            continue

        TX, TZ = np.meshgrid(x_range[ix_min:ix_max + 1],
                             z_range[iz_min:iz_max + 1], indexing='ij')
        inside = pad_path.contains_points(
            np.column_stack([TX.ravel(), TZ.ravel()])).reshape(n_ix, n_iz)

        iy_floor = max(0, int((pocket_floor_y - y_range[0]) / res))
        iy_top_pad = min(ny, int((pad_y_top - y_range[0]) / res) + 2)

        for di in range(n_ix):
            for dj in range(n_iz):
                if inside[di, dj]:
                    grid[ix_min + di, iy_floor:iy_top_pad, iz_min + dj] = 0
                    n_pocketed += 1
    print(f"  Pocketed {n_pocketed} columns for {len(NOTE_BY_INDEX)} pads")

    # Phase 5c: Carve groove channels into surface
    print("\nPhase 5c: Groove channels...")
    n_groove_carved = 0
    for note_key, top_verts in groove_top_surfaces.items():
        if len(top_verts) < 3:
            continue
        gxz = top_verts[:, [0, 2]]
        gy = top_verts[:, 1]
        gx_min, gz_min = gxz.min(axis=0)
        gx_max, gz_max = gxz.max(axis=0)
        ix_lo = max(0, int((gx_min - x_range[0]) / res) - 1)
        ix_hi = min(nx - 1, int((gx_max - x_range[0]) / res) + 2)
        iz_lo = max(0, int((gz_min - z_range[0]) / res) - 1)
        iz_hi = min(nz - 1, int((gz_max - z_range[0]) / res) + 2)
        n_gix = ix_hi - ix_lo + 1
        n_giz = iz_hi - iz_lo + 1
        if n_gix <= 0 or n_giz <= 0:
            continue
        GX, GZ = np.meshgrid(x_range[ix_lo:ix_hi + 1],
                              z_range[iz_lo:iz_hi + 1], indexing='ij')
        groove_y_map = griddata(gxz, gy, (GX, GZ), method='linear')
        try:
            ghull = ConvexHull(gxz)
            gpath = MplPath(gxz[ghull.vertices])
            g_inside = gpath.contains_points(
                np.column_stack([GX.ravel(), GZ.ravel()])).reshape(n_gix, n_giz)
        except Exception:
            continue
        for di in range(n_gix):
            for dj in range(n_giz):
                if not g_inside[di, dj] or np.isnan(groove_y_map[di, dj]):
                    continue
                groove_surf_y = groove_y_map[di, dj]
                iy_groove = int((groove_surf_y - y_range[0]) / res) + 1
                ix_abs = ix_lo + di
                iz_abs = iz_lo + dj
                # Only carve up to original surface (iy_top), not through drum wall
                iy_ceil = iy_top[ix_abs, iz_abs]
                if 0 <= iy_groove < iy_ceil:
                    n_groove_carved += int(grid[ix_abs, iy_groove:iy_ceil, iz_abs].sum())
                    grid[ix_abs, iy_groove:iy_ceil, iz_abs] = 0
    print(f"  Carved {n_groove_carved / 1e6:.1f}M groove voxels from {len(groove_top_surfaces)} grooves")

    # Phase 6: Carve wiring void
    # Cylinder: radius = drum_r - VOID_INSET, height = VOID_HEIGHT
    # Sits on top of base plate (Y from y_min + BASE_THICKNESS to
    # y_min + BASE_THICKNESS + VOID_HEIGHT)
    # Wiring void: cylinder from base plate top up to VOID_CLEARANCE below
    # the pan surface. Uses pre-computed void_top_map.
    print("\nPhase 6: Wiring void...")
    void_r = drum_r - VOID_INSET
    void_y_bot = y_min + BASE_THICKNESS
    iy_void_bot = max(0, int((void_y_bot - y_range[0]) / res))
    print(f"  R={void_r:.0f}mm, {VOID_CLEARANCE}mm below surface, base at Y={void_y_bot:.0f}")

    n_void = 0
    for ix in range(nx):
        for iz in range(nz):
            r = np.sqrt(x_range[ix]**2 + z_range[iz]**2)
            if r <= void_r:
                vt = void_top_map[ix, iz]
                if np.isnan(vt):
                    vt = rim_y - VOID_CLEARANCE
                iy_void_top = min(ny, int((vt - y_range[0]) / res) + 1)
                if iy_void_top > iy_void_bot:
                    before = grid[ix, iy_void_bot:iy_void_top, iz].sum()
                    grid[ix, iy_void_bot:iy_void_top, iz] = 0
                    n_void += before
    del void_top_map
    print(f"  Voided {n_void / 1e6:.1f}M voxels")

    # Keep col_top for cylinder surface clipping (needed by Tasks 9-10)
    del inside_drum, valid
    gc.collect()

    # ── Phase 6b: Cylinder + radial struts ──────────────────────────
    print("\nPhase 6b: Cylinder + radial struts...")
    iy_base_top_cyl = int((y_min + BASE_THICKNESS - y_range[0]) / res) + 1

    # Precompute XZ grids
    XX, ZZ = np.meshgrid(x_range, z_range, indexing='ij')
    RR_sq = XX**2 + ZZ**2
    RR = np.sqrt(RR_sq)
    iy_top_map = np.clip(((col_top - y_range[0]) / res).astype(np.int32), 0, ny)
    del col_top

    # Cylinder footprint (R=115..135)
    cyl_xz = (RR_sq >= CYL_INNER_R**2) & (RR_sq <= CYL_OUTER_R**2)

    # Strut footprints (20mm wide walls from center to drum edge)
    # Each strut only claims the "forward" half (radial >= 0) so that
    # opposite struts (e.g. 15° and 195°) don't fight over the same voxels.
    strut_tang = np.zeros((N_STRUTS, nx, nz), dtype=np.float32)
    strut_in = np.zeros((N_STRUTS, nx, nz), dtype=bool)
    for si, sa in enumerate(STRUT_ANGLES_RAD):
        cos_a = math.cos(sa)
        sin_a = math.sin(sa)
        strut_tang[si] = (-XX * sin_a + ZZ * cos_a).astype(np.float32)
        strut_radial = XX * cos_a + ZZ * sin_a
        strut_in[si] = (np.abs(strut_tang[si]) <= STRUT_HALF_W) & (strut_radial >= 0)
    strut_xz = strut_in.any(axis=0) & (RR <= drum_r)
    struct_xz = cyl_xz | strut_xz

    # Fill structural voxels into grid
    n_before = int(grid.sum())
    for ix in range(nx):
        for iz in range(nz):
            if struct_xz[ix, iz]:
                iy_top_s = iy_top_map[ix, iz]
                if iy_top_s > iy_base_top_cyl:
                    grid[ix, iy_base_top_cyl:iy_top_s, iz] = 1
    n_struct = int(grid.sum()) - n_before
    print(f"  Cylinder: R=[{CYL_INNER_R:.0f},{CYL_OUTER_R:.0f}]mm")
    print(f"  Struts: {N_STRUTS} x {STRUT_WALL:.0f}mm at {STRUT_ANGLES_DEG}")
    print(f"  Added {n_struct / 1e6:.1f}M structural voxels")

    # ── Phase 6c: Piece assignment (central + 6 outer) ───────────────
    # Assigns ALL grid columns (not just structure) so each piece includes
    # the drum body surface, pads, grooves, and structural support together.
    print("\nPhase 6c: Piece assignment...")
    # piece_xz: 0=outside drum, 1=central, 2..7=outer pieces 0..5
    piece_xz = np.zeros((nx, nz), dtype=np.uint8)
    ANGLES = np.arctan2(ZZ, XX)
    key_half_arc = KEY_WIDTH / (2.0 * CYL_SPLIT_R)
    within_drum = RR <= drum_r

    # Step 1: Central piece (R < split, within drum)
    inner_all = (RR < CYL_SPLIT_R) & within_drum
    piece_xz[inner_all] = 1

    # Step 2: Default sector for all outer columns
    outer_all = (RR >= CYL_SPLIT_R) & within_drum
    a_shifted = (ANGLES - STRUT_ANGLES_RAD[0]) % (2 * np.pi)
    default_sector = (a_shifted / (np.pi / 3)).astype(np.int32) % N_STRUTS
    piece_xz[outer_all] = (2 + default_sector[outer_all]).astype(np.uint8)

    # Step 3: Radial key override (near split radius, at strut angles)
    radial_key_zone = (RR >= CYL_SPLIT_R - KEY_DEPTH) & (RR < CYL_SPLIT_R)
    for si in range(N_STRUTS):
        sa = STRUT_ANGLES_RAD[si]
        angle_diff = (ANGLES - sa + np.pi) % (2 * np.pi) - np.pi
        in_key_arc = np.abs(angle_diff) <= key_half_arc
        key_mask = radial_key_zone & in_key_arc
        piece_xz[key_mask] = 2 + (si - 1) % N_STRUTS

    # Step 4: Side key/slot override on strut edges
    for si in range(N_STRUTS):
        in_this_strut = strut_in[si] & outer_all
        t = strut_tang[si]
        # At strut[si]: t < KEY_DEPTH -> sector (si-1)%6, else sector si
        key_side = in_this_strut & (t < KEY_DEPTH)
        piece_xz[key_side] = 2 + (si - 1) % N_STRUTS
        normal_side = in_this_strut & (t >= KEY_DEPTH)
        piece_xz[normal_side] = 2 + si

    n_central = int((piece_xz == 1).sum())
    n_outer = int((piece_xz >= 2).sum())
    print(f"  Central: {n_central} columns, Outer: {n_outer} columns")

    del (cyl_xz, strut_xz, strut_in, strut_tang, struct_xz, inner_all,
         outer_all, within_drum, a_shifted, default_sector, radial_key_zone,
         ANGLES, RR, RR_sq, XX, ZZ)
    gc.collect()

    # ── Phase 6d: Wiring holes ───────────────────────────────────────
    print("\nPhase 6d: Wiring holes (30mm dia)...")
    wire_base_y = y_min + BASE_THICKNESS
    wire_center_y = wire_base_y + WIRE_HOLE_CENTER_HEIGHT
    iy_wire_lo = max(0, int((wire_center_y - WIRE_HOLE_R - y_range[0]) / res) - 1)
    iy_wire_hi = min(ny, int((wire_center_y + WIRE_HOLE_R - y_range[0]) / res) + 2)
    n_wire = 0

    # Strut wiring holes: one per strut at R=STRUT_WIRE_R_POS, tangential axis
    for sa in STRUT_ANGLES_RAD:
        cos_a = math.cos(sa)
        sin_a = math.sin(sa)
        cx = STRUT_WIRE_R_POS * cos_a
        cz = STRUT_WIRE_R_POS * sin_a
        search_r = WIRE_HOLE_R + STRUT_HALF_W
        ix_lo = max(0, int((cx - search_r - x_range[0]) / res) - 1)
        ix_hi = min(nx - 1, int((cx + search_r - x_range[0]) / res) + 2)
        iz_lo = max(0, int((cz - search_r - z_range[0]) / res) - 1)
        iz_hi = min(nz - 1, int((cz + search_r - z_range[0]) / res) + 2)
        for ix in range(ix_lo, ix_hi + 1):
            dx = x_range[ix] - cx
            for iz in range(iz_lo, iz_hi + 1):
                dz = z_range[iz] - cz
                radial = dx * cos_a + dz * sin_a
                for iy in range(iy_wire_lo, iy_wire_hi):
                    vertical = y_range[iy] - wire_center_y
                    if radial * radial + vertical * vertical <= WIRE_HOLE_R ** 2:
                        if grid[ix, iy, iz] == 1:
                            grid[ix, iy, iz] = 0
                            n_wire += 1

    # Cylinder wiring holes: 6, at sector midpoints, radial axis
    for wa in WIRE_CYL_ANGLES_RAD:
        cos_a = math.cos(wa)
        sin_a = math.sin(wa)
        for ix in range(nx):
            for iz in range(nz):
                r_sq = x_range[ix]**2 + z_range[iz]**2
                if r_sq < CYL_INNER_R**2 or r_sq > CYL_OUTER_R**2:
                    continue
                tangential = -x_range[ix] * sin_a + z_range[iz] * cos_a
                if abs(tangential) > WIRE_HOLE_R + res:
                    continue
                for iy in range(iy_wire_lo, iy_wire_hi):
                    vertical = y_range[iy] - wire_center_y
                    if tangential * tangential + vertical * vertical <= WIRE_HOLE_R ** 2:
                        if grid[ix, iy, iz] == 1:
                            grid[ix, iy, iz] = 0
                            n_wire += 1
    print(f"  {n_wire} voxels removed ({N_STRUTS} strut + {N_STRUTS} cylinder holes)")

    # ── Phase 6e: Structural screw holes ─────────────────────────────
    print("\nPhase 6e: Structural screw holes...")
    struct_screw_positions = []

    def _drill_pilot(sx, sz, sy_base):
        """Drill one M3 pilot hole upward from base plate top."""
        iy_bot_s = int((sy_base - y_range[0]) / res)
        iy_top_s = min(ny, int((sy_base + SCREW_DEPTH - y_range[0]) / res) + 1)
        ix_min = max(0, int((sx - SCREW_PILOT_R - x_range[0]) / res) - 1)
        ix_max = min(nx - 1, int((sx + SCREW_PILOT_R - x_range[0]) / res) + 2)
        iz_min = max(0, int((sz - SCREW_PILOT_R - z_range[0]) / res) - 1)
        iz_max = min(nz - 1, int((sz + SCREW_PILOT_R - z_range[0]) / res) + 2)
        for ixx in range(ix_min, ix_max + 1):
            dx = x_range[ixx] - sx
            for izz in range(iz_min, iz_max + 1):
                dz = z_range[izz] - sz
                if dx * dx + dz * dz <= SCREW_PILOT_R ** 2:
                    grid[ixx, iy_bot_s:iy_top_s, izz] = 0

    # Cylinder screws: inner ring (R=120) and outer ring (R=130)
    sy_base = y_min + BASE_THICKNESS
    for ring_r in [CYL_SCREW_R_INNER, CYL_SCREW_R_OUTER]:
        for wa in WIRE_CYL_ANGLES_RAD:  # sector midpoints
            sx = ring_r * math.cos(wa)
            sz = ring_r * math.sin(wa)
            struct_screw_positions.append((sx, sy_base, sz))
            _drill_pilot(sx, sz, sy_base)

    # Strut screws: two rows per strut (5mm from each tangential edge)
    for sa in STRUT_ANGLES_RAD:
        cos_a = math.cos(sa)
        sin_a = math.sin(sa)
        r_pos = STRUT_SCREW_R_START
        while r_pos < drum_r - 10:
            for t_off in [-(STRUT_HALF_W - SCREW_EDGE_MAX),
                          STRUT_HALF_W - SCREW_EDGE_MAX]:
                sx = r_pos * cos_a + t_off * (-sin_a)
                sz = r_pos * sin_a + t_off * cos_a
                struct_screw_positions.append((sx, sy_base, sz))
                _drill_pilot(sx, sz, sy_base)
            r_pos += STRUT_SCREW_SPACING
    n_cyl_screws = 2 * CYL_SCREW_N
    n_strut_screws = len(struct_screw_positions) - n_cyl_screws
    print(f"  {n_cyl_screws} cylinder + {n_strut_screws} strut screws")

    del iy_top
    gc.collect()

    # Phase 7: Hardware holes — stop at base plate top (don't go through base)
    print("\nPhase 7: Hardware holes (stop at base)...")
    iy_base_top = int((y_min + BASE_THICKNESS - y_range[0]) / res) + 1
    for note, path in hw_shapes.items():
        bounds = path.get_extents()
        ix_min = max(0, int((bounds.x0 - x_range[0]) / res) - 1)
        ix_max = min(nx - 1, int((bounds.x1 - x_range[0]) / res) + 2)
        iz_min = max(0, int((bounds.y0 - z_range[0]) / res) - 1)
        iz_max = min(nz - 1, int((bounds.y1 - z_range[0]) / res) + 2)
        n_ix, n_iz = ix_max - ix_min + 1, iz_max - iz_min + 1
        if n_ix <= 0 or n_iz <= 0:
            continue
        TX, TZ = np.meshgrid(x_range[ix_min:ix_max + 1],
                             z_range[iz_min:iz_max + 1], indexing='ij')
        inside = path.contains_points(
            np.column_stack([TX.ravel(), TZ.ravel()])).reshape(n_ix, n_iz)
        for di in range(n_ix):
            for dj in range(n_iz):
                if inside[di, dj]:
                    # Clear from base top to surface (not through base)
                    grid[ix_min + di, iy_base_top:, iz_min + dj] = 0
    n_after = int(grid.sum())
    print(f"  {n_after / 1e6:.0f}M voxels")

    # Phase 7b: Bounding ellipse cavities (for central/inner push-cap mechanisms)
    print("\nPhase 7b: Bounding ellipse cavities...")
    from pathlib import Path as PathLib
    bc_path = 'data/notepads/bounding_ellipses.obj'
    n_bc_carved = 0
    if PathLib(bc_path).exists():
        bc_verts_all = []
        bc_objects = {}
        bc_current = None
        with open(bc_path) as bcf:
            for line in bcf:
                line = line.strip()
                if line.startswith('o '):
                    bc_current = line[2:]
                    bc_objects[bc_current] = {'v_start': len(bc_verts_all)}
                elif line.startswith('v ') and bc_current:
                    p = line.split()
                    bc_verts_all.append([float(p[1]), float(p[2]), float(p[3])])
        if bc_verts_all:
            bc_verts_all = np.array(bc_verts_all)
            # For each bounding ellipse, find XZ footprint and carve
            for bc_name, bc_info in bc_objects.items():
                v_start = bc_info['v_start']
                next_starts = [info['v_start'] for info in bc_objects.values()
                               if info['v_start'] > v_start]
                v_end = min(next_starts) if next_starts else len(bc_verts_all)
                bc_v = bc_verts_all[v_start:v_end]
                if len(bc_v) < 3:
                    continue

                # XZ footprint — recover ellipse axes via PCA
                bc_xz = bc_v[:, [0, 2]]
                bc_center_xz = bc_xz.mean(axis=0)
                centered = bc_xz - bc_center_xz
                cov = np.cov(centered.T)
                eigvals, eigvecs = np.linalg.eigh(cov)
                # Semi-axes from max projection onto each eigenvector
                proj0 = centered @ eigvecs[:, 0]
                proj1 = centered @ eigvecs[:, 1]
                semi_a = float(np.abs(proj0).max())  # axis 0
                semi_b = float(np.abs(proj1).max())  # axis 1
                ax0 = eigvecs[:, 0]  # unit vector in XZ
                ax1 = eigvecs[:, 1]

                bc_y_min = float(bc_v[:, 1].min())
                bc_y_max = float(bc_v[:, 1].max())

                # Bounding box from max semi-axis for voxel range
                bc_r_max = max(semi_a, semi_b)
                ix_min = max(0, int((bc_center_xz[0] - bc_r_max - x_range[0]) / res) - 1)
                ix_max = min(nx-1, int((bc_center_xz[0] + bc_r_max - x_range[0]) / res) + 2)
                iz_min = max(0, int((bc_center_xz[1] - bc_r_max - z_range[0]) / res) - 1)
                iz_max = min(nz-1, int((bc_center_xz[1] + bc_r_max - z_range[0]) / res) + 2)
                iy_bot = max(0, int((bc_y_min - y_range[0]) / res))
                iy_top = min(ny, int((bc_y_max - y_range[0]) / res) + 1)

                for ixx in range(ix_min, ix_max + 1):
                    dx = x_range[ixx] - bc_center_xz[0]
                    for izz in range(iz_min, iz_max + 1):
                        dz = z_range[izz] - bc_center_xz[1]
                        # Project onto ellipse principal axes
                        p0 = dx * ax0[0] + dz * ax0[1]
                        p1 = dx * ax1[0] + dz * ax1[1]
                        if semi_a > 0 and semi_b > 0:
                            if (p0 / semi_a) ** 2 + (p1 / semi_b) ** 2 <= 1.0:
                                before = int(grid[ixx, iy_bot:iy_top, izz].sum())
                                grid[ixx, iy_bot:iy_top, izz] = 0
                                n_bc_carved += before
        print(f"  Carved {n_bc_carved / 1e6:.1f}M voxels for {len(bc_objects)} bounding ellipses")
    else:
        print(f"  No bounding ellipses file found ({bc_path})")

    # Phase 8: Pad screw holes — pilot + countersink taper + plug bore
    # Profile from pocket floor downward:
    #   plug bore (CSINK_TOP_R, CSINK_PLUG_DEPTH) → taper (CSINK_TOP_R→PILOT_R, CSINK_TAPER_DEPTH) → pilot (PILOT_R)
    print("\nPhase 8: Pad screw holes + countersink + plug bore...")
    total_csink = CSINK_PLUG_DEPTH + CSINK_TAPER_DEPTH
    for hx, hy, hz in screw_holes:
        ix_min = max(0, int((hx - CSINK_TOP_R - x_range[0]) / res) - 1)
        ix_max = min(nx - 1, int((hx + CSINK_TOP_R - x_range[0]) / res) + 2)
        iz_min = max(0, int((hz - CSINK_TOP_R - z_range[0]) / res) - 1)
        iz_max = min(nz - 1, int((hz + CSINK_TOP_R - z_range[0]) / res) + 2)
        pocket_floor = hy - POCKET_DEPTH
        iy_top = int((pocket_floor - y_range[0]) / res) + 1
        iy_bot = max(0, int((pocket_floor - PILOT_DEPTH - y_range[0]) / res))
        for ix in range(ix_min, ix_max + 1):
            dx = x_range[ix] - hx
            for iz in range(iz_min, iz_max + 1):
                dz = z_range[iz] - hz
                r_sq = dx * dx + dz * dz
                if r_sq <= PILOT_R * PILOT_R:
                    # Cylindrical pilot hole — full depth
                    grid[ix, iy_bot:iy_top, iz] = 0
                elif r_sq <= CSINK_TOP_R * CSINK_TOP_R:
                    r = math.sqrt(r_sq)
                    # Plug bore: top CSINK_PLUG_DEPTH at full CSINK_TOP_R
                    plug_depth = CSINK_PLUG_DEPTH
                    # Taper: from CSINK_TOP_R down to PILOT_R
                    if r <= PILOT_R:
                        taper_extra = CSINK_TAPER_DEPTH
                    else:
                        taper_extra = CSINK_TAPER_DEPTH * (CSINK_TOP_R - r) / (CSINK_TOP_R - PILOT_R)
                    max_depth = plug_depth + taper_extra
                    iy_cone_bot = max(iy_bot, int((pocket_floor - max_depth - y_range[0]) / res))
                    grid[ix, iy_cone_bot:iy_top, iz] = 0
    print(f"  {len(screw_holes)} holes ({PILOT_R*2}mm pilot + {CSINK_TOP_R*2}mm csink + {CSINK_PLUG_DEPTH}mm bore)")

    # Phase 8b: Base attachment screw holes
    # M3 pilot holes in drum body (tapped), clearance holes in base plate
    # Arranged around the perimeter
    print("\nPhase 8b: Base attachment screws...")
    base_screw_r = drum_r - BASE_SCREW_INSET
    base_screw_positions = []
    for si in range(BASE_SCREW_N):
        angle = 2 * math.pi * si / BASE_SCREW_N
        sx = base_screw_r * math.cos(angle)
        sz = base_screw_r * math.sin(angle)
        # Y position: pilot hole goes UP into drum body from base plate top
        sy = y_min + BASE_THICKNESS  # starts at base plate top
        base_screw_positions.append((sx, sy, sz))

        # Pilot hole in drum body (above base plate)
        iy_bot_screw = int((sy - y_range[0]) / res)
        iy_top_screw = min(ny, int((sy + BASE_SCREW_DEPTH - y_range[0]) / res) + 1)
        ix_min = max(0, int((sx - BASE_SCREW_PILOT_R - x_range[0]) / res) - 1)
        ix_max = min(nx - 1, int((sx + BASE_SCREW_PILOT_R - x_range[0]) / res) + 2)
        iz_min = max(0, int((sz - BASE_SCREW_PILOT_R - z_range[0]) / res) - 1)
        iz_max = min(nz - 1, int((sz + BASE_SCREW_PILOT_R - z_range[0]) / res) + 2)
        for ix in range(ix_min, ix_max + 1):
            dx = x_range[ix] - sx
            for iz in range(iz_min, iz_max + 1):
                dz = z_range[iz] - sz
                if dx * dx + dz * dz <= BASE_SCREW_PILOT_R ** 2:
                    grid[ix, iy_bot_screw:iy_top_screw, iz] = 0

    n_final = int(grid.sum())
    print(f"  {BASE_SCREW_N} base screws at R={base_screw_r:.0f}mm")
    print(f"  Final body: {n_final / 1e6:.0f}M voxels")

    # Phase 9: Separate base plate
    print("\nPhase 9: Separate base plate...")
    iy_base_top_sep = int((y_min + BASE_THICKNESS - y_range[0]) / res) + 1
    base_grid = grid[:, :iy_base_top_sep, :].copy()
    grid[:, :iy_base_top_sep, :] = 0  # remove base from main body
    print(f"  Base: Y < {y_min + BASE_THICKNESS:.0f}mm, "
          f"{int(base_grid.sum()) / 1e6:.0f}M voxels")
    print(f"  Body without base: {int(grid.sum()) / 1e6:.0f}M voxels")

    # Drill M3 clearance holes through base plate for attachment screws
    print("  Drilling base clearance holes...")
    for sx, sy, sz in base_screw_positions:
        ix_min = max(0, int((sx - BASE_SCREW_CLEAR_R - x_range[0]) / res) - 1)
        ix_max = min(nx - 1, int((sx + BASE_SCREW_CLEAR_R - x_range[0]) / res) + 2)
        iz_min = max(0, int((sz - BASE_SCREW_CLEAR_R - z_range[0]) / res) - 1)
        iz_max = min(nz - 1, int((sz + BASE_SCREW_CLEAR_R - z_range[0]) / res) + 2)
        for ix in range(ix_min, ix_max + 1):
            dx = x_range[ix] - sx
            for iz in range(iz_min, iz_max + 1):
                dz = z_range[iz] - sz
                if dx * dx + dz * dz <= BASE_SCREW_CLEAR_R ** 2:
                    base_grid[ix, :, iz] = 0  # through-hole in base
    # Also drill clearance holes for structural screws
    print("  Drilling structural screw clearance holes in base...")
    for sx, sy, sz in struct_screw_positions:
        ix_min = max(0, int((sx - SCREW_CLEAR_R - x_range[0]) / res) - 1)
        ix_max = min(nx - 1, int((sx + SCREW_CLEAR_R - x_range[0]) / res) + 2)
        iz_min = max(0, int((sz - SCREW_CLEAR_R - z_range[0]) / res) - 1)
        iz_max = min(nz - 1, int((sz + SCREW_CLEAR_R - z_range[0]) / res) + 2)
        for ix in range(ix_min, ix_max + 1):
            dx = x_range[ix] - sx
            for iz in range(iz_min, iz_max + 1):
                dz = z_range[iz] - sz
                if dx * dx + dz * dz <= SCREW_CLEAR_R ** 2:
                    base_grid[ix, :, iz] = 0
    print(f"  Base after all holes: {int(base_grid.sum()) / 1e6:.0f}M voxels")

    if save_grid or grid_only:
        gpath = f'data/pan_body_{res}mm_grid.npz'
        print(f"\n  Saving grids to {gpath}...")
        np.savez_compressed(gpath, body=grid, base=base_grid,
                            piece_xz=piece_xz, iy_top_map=iy_top_map,
                            iy_base_top_cyl=np.array([iy_base_top_cyl]),
                            x_range=x_range, y_range=y_range,
                            z_range=z_range, res=np.array([res]))
        print("  Saved.")
        if grid_only:
            print("\n  --grid-only: skipping mesh extraction.")
            print("  Run extract_pieces.py to extract meshes from the saved grid.")
            print("\nDone!")
            return

    # Phase 10: Extract meshes via marching cubes
    tag = f'_{res}mm' if res != 0.5 else ''
    if use_sdf:
        tag += '_sdf'

    def _extract_mesh(voxel_grid, label):
        """Extract mesh from voxel grid via marching cubes (with optional SDF)."""
        padded = np.pad(voxel_grid, 1, mode='constant', constant_values=0)
        if use_sdf:
            v, f = _sdf_extract(padded, res, sigma)
        else:
            v, f, _, _ = marching_cubes(padded, level=0.5, spacing=(res, res, res))
        del padded
        v[:, 0] += x_range[0] - res
        v[:, 1] += y_range[0] - res
        v[:, 2] += z_range[0] - res
        print(f"  {label}: {len(v)}v, {len(f)}f")
        return v, f

    def _write_stl(path, label, verts, faces):
        with open(path, 'wb') as f:
            f.write(label.encode().ljust(80, b'\0'))
            f.write(struct.pack('<I', len(faces)))
            for tri in faces:
                v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
                n = np.cross(v1 - v0, v2 - v0)
                nl = np.linalg.norm(n)
                if nl > 0:
                    n /= nl
                f.write(struct.pack('<3f', *n))
                f.write(struct.pack('<3f', *v0))
                f.write(struct.pack('<3f', *v1))
                f.write(struct.pack('<3f', *v2))
                f.write(struct.pack('<H', 0))

    def _write_obj(path, label, verts, faces, obj_name="Mesh"):
        with open(path, 'w') as f:
            f.write(f"# {label}\n# Units: mm\n\n")
            f.write(f"o {obj_name}\n")
            for v in verts:
                f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
            f.write("\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    # Phase 10: Extract pieces — each includes drum surface + structure
    # piece_xz covers the full grid, so each piece gets its sector of the
    # playing surface, pads, grooves, AND structural support together.
    print("\nPhase 10: Extract meshes...")

    # Phase 10a: Central piece (R < 125mm: inner surface + cylinder hub + struts)
    print("\nPhase 10a: Central piece mesh...")
    mask = (piece_xz[:, np.newaxis, :] == 1)
    central_grid = np.where(mask, grid, np.uint8(0))
    del mask
    verts_central, faces_central = _extract_mesh(central_grid, "CentralPiece")
    del central_grid; gc.collect()

    # Phase 10b: Outer pieces (6 x 60deg wedge with pads + struts)
    outer_meshes = []
    for sector in range(N_STRUTS):
        piece_id = 2 + sector
        print(f"\nPhase 10b.{sector}: Outer piece {sector}...")
        mask = (piece_xz[:, np.newaxis, :] == piece_id)
        piece_grid = np.where(mask, grid, np.uint8(0))
        del mask
        v, f = _extract_mesh(piece_grid, f"OuterPiece_{sector}")
        outer_meshes.append((v, f))
        del piece_grid; gc.collect()

    del piece_xz, iy_top_map, grid; gc.collect()

    # Phase 10c: Base plate
    print("\nPhase 10c: Base plate mesh...")
    verts_base, faces_base = _extract_mesh(base_grid, "BasePlate")
    del base_grid; gc.collect()

    # Phase 12: Export all components
    print(f"\nPhase 12: Export (pan_body{tag})...")

    # Combined OBJ with all named objects
    with open(f'data/pan_body{tag}.obj', 'w') as f:
        f.write("# Pan pieces + base plate\n# Units: mm\n\n")
        vert_off = 0

        f.write("o CentralPiece\n")
        for v in verts_central:
            f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        for face in faces_central:
            f.write(f"f {face[0]+1+vert_off} {face[1]+1+vert_off} {face[2]+1+vert_off}\n")
        vert_off += len(verts_central)

        for sector in range(N_STRUTS):
            v_out, f_out = outer_meshes[sector]
            f.write(f"\no OuterPiece_{sector}\n")
            for v in v_out:
                f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
            for face in f_out:
                f.write(f"f {face[0]+1+vert_off} {face[1]+1+vert_off} {face[2]+1+vert_off}\n")
            vert_off += len(v_out)

        f.write(f"\no BasePlate\n")
        for v in verts_base:
            f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        for face in faces_base:
            f.write(f"f {face[0]+1+vert_off} {face[1]+1+vert_off} {face[2]+1+vert_off}\n")
    print(f"  pan_body{tag}.obj (CentralPiece + OuterPiece_0..5 + BasePlate)")

    # Separate OBJ files
    _write_obj(f'data/pan_base{tag}.obj', 'Base plate', verts_base, faces_base, 'BasePlate')
    print(f"  pan_base{tag}.obj")

    _write_obj(f'data/pan_central_piece{tag}.obj', 'Central piece',
               verts_central, faces_central, 'CentralPiece')
    print(f"  pan_central_piece{tag}.obj")

    for sector in range(N_STRUTS):
        v_out, f_out = outer_meshes[sector]
        _write_obj(f'data/pan_outer_piece_{sector}{tag}.obj',
                   f'Outer piece {sector}', v_out, f_out, f'OuterPiece_{sector}')
        print(f"  pan_outer_piece_{sector}{tag}.obj")

    # STL files
    _write_stl(f'data/pan_base{tag}.stl', 'STL base plate', verts_base, faces_base)
    print(f"  pan_base{tag}.stl ({len(faces_base)} tri)")

    _write_stl(f'data/pan_central_piece{tag}.stl', 'STL central piece',
               verts_central, faces_central)
    print(f"  pan_central_piece{tag}.stl ({len(faces_central)} tri)")

    for sector in range(N_STRUTS):
        v_out, f_out = outer_meshes[sector]
        _write_stl(f'data/pan_outer_piece_{sector}{tag}.stl',
                   f'STL outer piece {sector}', v_out, f_out)
        print(f"  pan_outer_piece_{sector}{tag}.stl ({len(f_out)} tri)")

    # Combined STL (all pieces + base)
    all_meshes = [(verts_central, faces_central)]
    for v_out, f_out in outer_meshes:
        all_meshes.append((v_out, f_out))
    all_meshes.append((verts_base, faces_base))
    all_verts_list, all_faces_list = [], []
    vert_off = 0
    for v, f in all_meshes:
        all_verts_list.append(v)
        all_faces_list.extend([[tri[0]+vert_off, tri[1]+vert_off, tri[2]+vert_off]
                               for tri in f])
        vert_off += len(v)
    all_verts = np.vstack(all_verts_list)
    _write_stl(f'data/pan_body{tag}.stl', 'STL pan body', all_verts, all_faces_list)
    print(f"  pan_body{tag}.stl ({len(all_faces_list)} tri, combined)")
    print("\nDone!")


if __name__ == "__main__":
    main()
