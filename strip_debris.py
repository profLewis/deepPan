#!/usr/bin/env python3
"""
Strip tiny disconnected mesh fragments from an STL, keeping only the largest
connected component. Voxel remesh sometimes leaves behind sub-voxel isolated
blobs that look like holes/debris in viewers but aren't part of the main shell.

Usage:
    python strip_debris.py [--in=PATH] [--out=PATH] [--min-faces=100]
"""
import argparse
import struct
import numpy as np
import trimesh


def write_stl_binary(path, verts, faces):
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    tri_v = verts[faces]
    n0 = np.cross(tri_v[:, 1] - tri_v[:, 0], tri_v[:, 2] - tri_v[:, 0])
    norms = np.linalg.norm(n0, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    n0 = (n0 / norms).astype(np.float32)
    with open(path, 'wb') as f:
        f.write(b'\0' * 80)
        f.write(struct.pack('<I', len(faces)))
        for ni, tri in zip(n0, tri_v):
            f.write(struct.pack('<3f', *ni))
            f.write(struct.pack('<3f', *tri[0]))
            f.write(struct.pack('<3f', *tri[1]))
            f.write(struct.pack('<3f', *tri[2]))
            f.write(b'\0\0')


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp",
                   default="data/quarters/pan_assembly_remesh.stl")
    p.add_argument("--out",
                   default="data/quarters/pan_assembly_remesh.stl")
    p.add_argument("--min-faces", type=int, default=1000,
                   help="components with fewer than this many faces are discarded")
    args = p.parse_args()

    print(f"Loading {args.inp} ...")
    m = trimesh.load(args.inp)
    print(f"  {len(m.vertices):,} verts, {len(m.faces):,} faces  "
          f"watertight={m.is_watertight}  Euler={m.euler_number}")

    print(f"Splitting into connected components ...")
    parts = m.split(only_watertight=False)
    parts_sorted = sorted(parts, key=lambda p: -len(p.faces))
    print(f"  {len(parts)} components, largest = {len(parts_sorted[0].faces):,} faces")

    keep = [p for p in parts_sorted if len(p.faces) >= args.min_faces]
    drop = [p for p in parts_sorted if len(p.faces) < args.min_faces]
    print(f"  Keeping {len(keep)} (>= {args.min_faces} faces), "
          f"discarding {len(drop)} ({sum(len(p.faces) for p in drop)} tris total)")

    # Concatenate kept components
    cleaned = trimesh.util.concatenate(keep) if len(keep) > 1 else keep[0]
    print(f"  Result: {len(cleaned.vertices):,} verts, {len(cleaned.faces):,} faces  "
          f"watertight={cleaned.is_watertight}  Euler={cleaned.euler_number}")

    write_stl_binary(args.out, cleaned.vertices, cleaned.faces)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
