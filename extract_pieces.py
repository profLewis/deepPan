#!/usr/bin/env python3
"""
Extract structural pieces from a saved pan body voxel grid.

Loads the .npz saved by generate_pan_body.py --save-grid, crops each piece
to its bounding box, runs SDF marching cubes on the cropped volume, and
exports OBJ + STL files.

This avoids running SDF on the full grid (which can exceed available RAM at
fine resolutions like 0.2mm).

Usage:
    python extract_pieces.py [--grid=data/pan_body_0.2mm_grid.npz]
                             [--sigma=0.5] [--no-sdf]
"""

import numpy as np
import struct
import gc
import sys

from skimage.measure import marching_cubes
from generate_cylinders import N_STRUTS


def _sdf_extract(padded, res, sigma):
    """SDF-based mesh extraction from padded binary grid."""
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    shape = padded.shape
    n_voxels = padded.size
    pb = padded.view(np.bool_)

    if n_voxels < 2_000_000_000:
        print(f"    SDF in-memory ({n_voxels / 1e6:.0f}M voxels)...")
        print("    interior...")
        sdf = distance_transform_edt(pb).astype(np.float32)
        print("    exterior...")
        np.logical_not(pb, out=pb)
        sdf -= distance_transform_edt(pb).astype(np.float32)
    else:
        import os, tempfile
        td = tempfile.mkdtemp(prefix='pan_sdf_')
        pin = os.path.join(td, 'in.f32')
        pex = os.path.join(td, 'ex.f32')

        print(f"    SDF disk-spill ({n_voxels / 1e9:.1f}B voxels)...")
        print("    interior...")
        d = distance_transform_edt(pb)
        with open(pin, 'wb') as f:
            for i in range(shape[0]):
                f.write(d[i].astype(np.float32).tobytes())
        del d; gc.collect()

        print("    exterior...")
        np.logical_not(pb, out=pb)
        d = distance_transform_edt(pb)
        with open(pex, 'wb') as f:
            for i in range(shape[0]):
                f.write(d[i].astype(np.float32).tobytes())
        del d; gc.collect()

        print("    combining from disk...")
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
        print(f"    Gaussian sigma={sigma:.1f}...")
        gaussian_filter(sdf, sigma=sigma, output=sdf)

    print("    marching cubes...")
    verts, faces, _, _ = marching_cubes(sdf, level=0.0, spacing=(res, res, res))
    del sdf; gc.collect()
    return verts, faces


def _crop_and_extract(grid_full, x_range, y_range, z_range, res, use_sdf,
                      sigma, label):
    """Crop a grid to its non-zero bounding box, extract mesh, return
    world-space vertices and faces."""
    # Find bounding box of non-zero voxels
    nz_x, nz_y, nz_z = np.nonzero(grid_full)
    if len(nz_x) == 0:
        print(f"  {label}: empty — skipping")
        return None, None
    ix0, ix1 = int(nz_x.min()), int(nz_x.max()) + 1
    iy0, iy1 = int(nz_y.min()), int(nz_y.max()) + 1
    iz0, iz1 = int(nz_z.min()), int(nz_z.max()) + 1
    del nz_x, nz_y, nz_z

    crop = grid_full[ix0:ix1, iy0:iy1, iz0:iz1]
    n_vox = int(crop.sum())
    shape = crop.shape
    print(f"  {label}: crop [{ix0}:{ix1}, {iy0}:{iy1}, {iz0}:{iz1}] = "
          f"{shape[0]}x{shape[1]}x{shape[2]} "
          f"({shape[0]*shape[1]*shape[2]/1e6:.0f}M voxels, {n_vox/1e6:.1f}M solid)")

    # Pad and extract
    padded = np.pad(crop, 1, mode='constant', constant_values=0)
    del crop; gc.collect()

    if use_sdf:
        verts, faces = _sdf_extract(padded, res, sigma)
    else:
        verts, faces, _, _ = marching_cubes(padded, level=0.5,
                                            spacing=(res, res, res))
    del padded; gc.collect()

    # Offset to world coordinates (crop origin + padding offset)
    verts[:, 0] += x_range[ix0] - res
    verts[:, 1] += y_range[iy0] - res
    verts[:, 2] += z_range[iz0] - res
    print(f"    {len(verts)}v, {len(faces)}f")
    return verts, faces


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


def main():
    # Parse args
    grid_path = None
    sigma = 0.5
    use_sdf = True
    for arg in sys.argv[1:]:
        if arg.startswith('--grid='):
            grid_path = arg.split('=', 1)[1]
        elif arg.startswith('--sigma='):
            sigma = float(arg.split('=')[1])
        elif arg == '--no-sdf':
            use_sdf = False

    # Find grid file if not specified
    if grid_path is None:
        import glob
        candidates = sorted(glob.glob('data/pan_body_*mm_grid.npz'))
        if not candidates:
            print("No grid file found. Run generate_pan_body.py --save-grid first.")
            sys.exit(1)
        grid_path = candidates[-1]  # newest / finest resolution

    print("=" * 60)
    print(f"Extract Pieces from {grid_path}")
    if use_sdf:
        print(f"  SDF sigma={sigma:.1f}")
    else:
        print("  Binary marching cubes (no SDF)")
    print("=" * 60)

    # Load
    print("\nLoading grid...")
    d = np.load(grid_path)
    grid = d['body']
    base_grid = d['base']
    piece_xz = d['piece_xz']
    x_range = d['x_range']
    y_range = d['y_range']
    z_range = d['z_range']
    res = float(d['res'][0])
    nx, ny, nz = grid.shape
    print(f"  Grid: {nx}x{ny}x{nz} = {nx*ny*nz/1e6:.0f}M voxels, res={res}mm")
    print(f"  piece_xz: {piece_xz.shape}")

    tag = f'_{res}mm' if res != 0.5 else ''
    if use_sdf:
        tag += '_sdf'

    # ── Extract base plate ────────────────────────────────────────
    print("\n── Base plate ──")
    verts_base, faces_base = _crop_and_extract(
        base_grid, x_range, y_range, z_range, res, use_sdf, sigma, "BasePlate")
    del base_grid; gc.collect()
    if verts_base is not None:
        _write_obj(f'data/pan_base{tag}.obj', 'Base plate',
                   verts_base, faces_base, 'BasePlate')
        _write_stl(f'data/pan_base{tag}.stl', 'STL base plate',
                   verts_base, faces_base)
        print(f"  -> pan_base{tag}.obj/stl ({len(faces_base)} tri)")

    # ── Extract central piece (R<125: surface + structure) ────────
    print("\n── Central piece ──")
    mask = (piece_xz[:, np.newaxis, :] == 1)
    central_grid = np.where(mask, grid, np.uint8(0))
    del mask; gc.collect()
    verts_central, faces_central = _crop_and_extract(
        central_grid, x_range, y_range, z_range, res, use_sdf, sigma,
        "CentralPiece")
    del central_grid; gc.collect()
    if verts_central is not None:
        _write_obj(f'data/pan_central_piece{tag}.obj', 'Central piece',
                   verts_central, faces_central, 'CentralPiece')
        _write_stl(f'data/pan_central_piece{tag}.stl', 'STL central piece',
                   verts_central, faces_central)
        print(f"  -> pan_central_piece{tag}.obj/stl ({len(faces_central)} tri)")

    # ── Extract outer pieces (6 x 60deg: pads + surface + struts) ─
    outer_meshes = []
    for sector in range(N_STRUTS):
        piece_id = 2 + sector
        print(f"\n── Outer piece {sector} ──")
        mask = (piece_xz[:, np.newaxis, :] == piece_id)
        piece_grid = np.where(mask, grid, np.uint8(0))
        del mask; gc.collect()
        v, f = _crop_and_extract(
            piece_grid, x_range, y_range, z_range, res, use_sdf, sigma,
            f"OuterPiece_{sector}")
        del piece_grid; gc.collect()
        outer_meshes.append((v, f))
        if v is not None:
            _write_obj(f'data/pan_outer_piece_{sector}{tag}.obj',
                       f'Outer piece {sector}', v, f, f'OuterPiece_{sector}')
            _write_stl(f'data/pan_outer_piece_{sector}{tag}.stl',
                       f'STL outer piece {sector}', v, f)
            print(f"  -> pan_outer_piece_{sector}{tag}.obj/stl ({len(f)} tri)")

    del grid, piece_xz; gc.collect()

    # ── Combined OBJ ──────────────────────────────────────────────
    print(f"\n── Combined OBJ: pan_body{tag}.obj ──")
    all_pieces = []
    if verts_central is not None:
        all_pieces.append(('CentralPiece', verts_central, faces_central))
    for sector in range(N_STRUTS):
        v, f = outer_meshes[sector]
        if v is not None:
            all_pieces.append((f'OuterPiece_{sector}', v, f))
    if verts_base is not None:
        all_pieces.append(('BasePlate', verts_base, faces_base))

    with open(f'data/pan_body{tag}.obj', 'w') as fout:
        fout.write("# Pan pieces + base plate\n# Units: mm\n\n")
        vert_off = 0
        for obj_name, verts, faces in all_pieces:
            fout.write(f"o {obj_name}\n")
            for v in verts:
                fout.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
            for face in faces:
                fout.write(f"f {face[0]+1+vert_off} {face[1]+1+vert_off} "
                           f"{face[2]+1+vert_off}\n")
            vert_off += len(verts)
            fout.write("\n")
    total_tri = sum(len(f) for _, _, f in all_pieces)
    print(f"  {len(all_pieces)} objects, {total_tri} tri total")

    # ── Combined STL ──────────────────────────────────────────────
    if all_pieces:
        all_verts_list, all_faces_list = [], []
        vert_off = 0
        for _, v, f in all_pieces:
            all_verts_list.append(v)
            all_faces_list.extend(
                [[tri[0]+vert_off, tri[1]+vert_off, tri[2]+vert_off]
                 for tri in f])
            vert_off += len(v)
        all_verts = np.vstack(all_verts_list)
        _write_stl(f'data/pan_body{tag}.stl', 'STL pan body',
                   all_verts, all_faces_list)
        print(f"  pan_body{tag}.stl ({len(all_faces_list)} tri, combined)")

    print("\nDone!")


if __name__ == "__main__":
    main()
