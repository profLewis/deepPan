#!/usr/bin/env python3
"""
Deform the pantry surface by pushing it down where assembly objects sit.

Processes each assembly body individually (memory-safe), builds a min-Y
heightmap, then deforms the subdivided pantry mesh. Uses edge-preserving
median filter for smoothing.

Usage:
    python push_assembly_into_pantry.py [--res=0.25] [--tol=0.3] [--subdiv=3] [--adaptive=3]
"""

import sys, gc
import numpy as np
import trimesh
from scipy.ndimage import median_filter, minimum_filter


RES = 0.25
TOL = 0.3
SUBDIV = 3
ADAPTIVE = 3
PANTRY = "data/pantry.stl"
ASSEMBLY = "data/notepads/assembly_view.stl"
OUTPUT = "data/pantry_with_pockets_v2.stl"


def load_mesh(path):
    m = trimesh.load(path)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(list(m.geometry.values()))
    return m


def build_heightmap_per_body(assembly, x_range, z_range, res):
    """Build min-Y heightmap by processing each assembly body individually.
    Each body is made solid (if possible), then sampled. Memory-safe."""
    nx, nz = len(x_range), len(z_range)
    x0, z0 = x_range[0], z_range[0]
    min_y = np.full((nx, nz), np.nan, dtype=np.float32)

    bodies = assembly.split(only_watertight=False)
    bodies = [b for b in bodies if len(b.faces) >= 6]
    print(f"    {len(bodies)} bodies to process")

    for bi, body in enumerate(bodies):
        # Make body solid
        trimesh.repair.fix_normals(body)
        trimesh.repair.fill_holes(body)
        if not body.is_volume:
            try:
                bv = body.voxelized(max(res, 0.3))
                body = bv.marching_cubes
                body.apply_transform(bv.transform)
                trimesh.repair.fix_normals(body)
            except Exception:
                pass  # use as-is

        # Sample surface
        n_samp = max(1000, len(body.faces) * 3)
        try:
            pts, _ = trimesh.sample.sample_surface(body, min(n_samp, 50000))
            all_pts = np.vstack([pts, body.vertices])
        except Exception:
            all_pts = body.vertices

        # Scatter to grid
        ix = np.round((all_pts[:, 0] - x0) / res).astype(np.int32)
        iz = np.round((all_pts[:, 2] - z0) / res).astype(np.int32)
        valid = (ix >= 0) & (ix < nx) & (iz >= 0) & (iz < nz)
        if valid.any():
            ix_v, iz_v = ix[valid], iz[valid]
            yv = all_pts[valid, 1].astype(np.float32)
            for k in range(len(ix_v)):
                i, j, y = ix_v[k], iz_v[k], yv[k]
                if np.isnan(min_y[i, j]) or y < min_y[i, j]:
                    min_y[i, j] = y

        if (bi + 1) % 50 == 0:
            print(f"      {bi+1}/{len(bodies)}...", flush=True)
        del body, all_pts
        gc.collect()

    # Fill small gaps
    filled = minimum_filter(min_y, size=3)
    gaps = np.isnan(min_y) & ~np.isnan(filled)
    min_y[gaps] = filled[gaps]

    n_covered = (~np.isnan(min_y)).sum()
    print(f"    {n_covered/1e3:.0f}K cells covered ({n_covered/(nx*nz)*100:.1f}%)")
    return min_y


def deform_vertices(verts, asm_target_y, x_range, z_range, res):
    """Push vertices down where assembly exists."""
    nx, nz = len(x_range), len(z_range)
    x0, z0 = x_range[0], z_range[0]
    new_verts = verts.copy()

    ix = np.round((verts[:, 0] - x0) / res).astype(np.int32)
    iz = np.round((verts[:, 2] - z0) / res).astype(np.int32)
    in_bounds = (ix >= 0) & (ix < nx) & (iz >= 0) & (iz < nz)

    target_y = np.full(len(verts), np.nan)
    target_y[in_bounds] = asm_target_y[ix[in_bounds], iz[in_bounds]]

    push_mask = in_bounds & ~np.isnan(target_y) & (verts[:, 1] > target_y)
    new_verts[push_mask, 1] = target_y[push_mask]
    return new_verts, push_mask


def adaptive_split(verts, faces, push_mask):
    """Split faces straddling pocket boundaries."""
    new_verts_list = list(verts)
    new_faces = []
    edge_midpoints = {}

    def get_mid(v1, v2):
        key = (min(v1, v2), max(v1, v2))
        if key not in edge_midpoints:
            edge_midpoints[key] = len(new_verts_list)
            new_verts_list.append((verts[v1] + verts[v2]) / 2.0)
        return edge_midpoints[key]

    n_split = 0
    for a, b, c in faces:
        if (push_mask[a] + push_mask[b] + push_mask[c]) in (0, 3):
            new_faces.append([a, b, c])
        else:
            mab, mbc, mca = get_mid(a, b), get_mid(b, c), get_mid(c, a)
            new_faces.extend([[a,mab,mca],[b,mbc,mab],[c,mca,mbc],[mab,mbc,mca]])
            n_split += 1

    return np.array(new_verts_list), np.array(new_faces), n_split


def smooth_median(orig_y, new_y, verts, x_range, z_range, res, radius_mm=1.0):
    """Edge-preserving median filter on the displacement field."""
    nx, nz = len(x_range), len(z_range)
    x0, z0 = x_range[0], z_range[0]

    disp_grid = np.zeros((nx, nz), dtype=np.float32)
    ix = np.round((verts[:, 0] - x0) / res).astype(np.int32)
    iz = np.round((verts[:, 2] - z0) / res).astype(np.int32)
    in_bounds = (ix >= 0) & (ix < nx) & (iz >= 0) & (iz < nz)
    displacement = new_y - orig_y

    for vi in range(len(verts)):
        if in_bounds[vi] and displacement[vi] < -0.001:
            d = displacement[vi]
            if d < disp_grid[ix[vi], iz[vi]]:
                disp_grid[ix[vi], iz[vi]] = d

    # Median filter preserves edges (pocket walls stay sharp)
    k = max(3, int(radius_mm / res) * 2 + 1)
    disp_smooth = median_filter(disp_grid, size=k)

    result_y = new_y.copy()
    for vi in range(len(verts)):
        if in_bounds[vi]:
            d = disp_smooth[ix[vi], iz[vi]]
            if d < -0.01:
                result_y[vi] = min(result_y[vi], orig_y[vi] + d)
    return result_y


def main():
    global RES, TOL, SUBDIV, ADAPTIVE
    for arg in sys.argv[1:]:
        if arg.startswith('--res='): RES = float(arg.split('=')[1])
        elif arg.startswith('--tol='): TOL = float(arg.split('=')[1])
        elif arg.startswith('--subdiv='): SUBDIV = int(arg.split('=')[1])
        elif arg.startswith('--adaptive='): ADAPTIVE = int(arg.split('=')[1])

    print("=" * 60)
    print("Push assembly into pantry (plasticine deformation)")
    print(f"  Grid={RES}mm  Tol={TOL}mm  Subdiv={SUBDIV}  Adaptive={ADAPTIVE}")
    print("=" * 60)

    pantry = load_mesh(PANTRY)
    print(f"Pantry: {len(pantry.faces)/1e3:.0f}K faces")
    assembly = load_mesh(ASSEMBLY)
    print(f"Assembly: {len(assembly.faces)/1e3:.0f}K faces")

    # Global subdivision
    print(f"\nSubdividing pantry ({SUBDIV} levels)...")
    v, f = pantry.vertices, pantry.faces
    for i in range(SUBDIV):
        v, f = trimesh.remesh.subdivide(v, f)
        print(f"  {i+1}: {len(f)/1e6:.1f}M faces, {len(v)/1e6:.1f}M verts")
    del pantry; gc.collect()

    # XZ grid
    margin = 2.0
    x_range = np.arange(v[:, 0].min() - margin, v[:, 0].max() + margin + RES, RES)
    z_range = np.arange(v[:, 2].min() - margin, v[:, 2].max() + margin + RES, RES)
    print(f"\nXZ grid: {len(x_range)}x{len(z_range)}")

    # Build heightmap from solid bodies (one at a time, memory-safe)
    print("\nBuilding heightmap (per-body, solid)...")
    asm_min_y = build_heightmap_per_body(assembly, x_range, z_range, RES)
    asm_target_y = asm_min_y - TOL
    del assembly; gc.collect()

    # Iterative deform + adaptive split
    faces = np.array(f)
    verts = np.array(v)
    del v, f; gc.collect()
    orig_y = verts[:, 1].copy()

    for it in range(1 + ADAPTIVE):
        new_verts, push_mask = deform_vertices(verts, asm_target_y, x_range, z_range, RES)
        n_pushed = push_mask.sum()
        print(f"\n  Pass {it}: {n_pushed} pushed, {len(faces)/1e6:.1f}M faces")

        if it < ADAPTIVE:
            verts, faces, n_split = adaptive_split(new_verts, faces, push_mask)
            print(f"    Split {n_split} boundary faces → {len(faces)/1e6:.1f}M faces")
            n_new = len(verts) - len(orig_y)
            if n_new > 0:
                orig_y = np.concatenate([orig_y, verts[len(orig_y):, 1]])
            push_mask = np.zeros(len(verts), dtype=bool)
        else:
            verts = new_verts

    # Edge-preserving smooth
    print(f"\n  Median filter (edge-preserving)...", end="", flush=True)
    verts[:, 1] = smooth_median(orig_y, verts[:, 1], verts, x_range, z_range, RES)
    print(" done")

    result = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    trimesh.repair.fix_normals(result)

    y_diff = orig_y - verts[:, 1]
    pushed = y_diff > 0.01
    print(f"\nResult: {len(faces)/1e6:.1f}M faces")
    print(f"  Pushed: {pushed.sum()}, Max: {y_diff.max():.1f}mm, Mean: {y_diff[pushed].mean():.1f}mm")

    print(f"\nSaving {OUTPUT}...")
    result.export(OUTPUT)
    print("Done!")


if __name__ == "__main__":
    main()
