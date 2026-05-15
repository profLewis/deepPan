#!/usr/bin/env python3
"""
Replace each per-pad groove ring with a simple oriented ellipsoid:
  - Fit a best-fit plane to the groove face (SVD on its verts)
  - Build an ellipsoid whose 2 in-plane axes match the groove's bbox in-plane,
    its 3rd axis matches the groove's perpendicular extent (or a 5mm depth
    floor — matches solidify_grooves.py thickness).
  - Orient the ellipsoid to match the plane.
  - Extend planimetrically (scale X and Z components by (1+XZ_MARGIN)).

These ellipsoids replace the complex groove ring solids for the purpose of
filling the body's gaps at groove-pan junctions. Boolean-unioning them with
pan_holes_solid produces a clean closed body without seams at the grooves.

Outputs (one combined OBJ/STL per ring set):
    pipeline_output/groove_ellipsoids_central.{obj,stl}
    pipeline_output/groove_ellipsoids_inner.{obj,stl}
    pipeline_output/groove_ellipsoids_outer.{obj,stl}
"""
from pathlib import Path
import numpy as np
import trimesh

XZ_MARGIN = 0.15  # planimetric extension (X and Z scaled by 1 + this)
DEPTH_FLOOR = 5.0  # mm — minimum perpendicular thickness (Y of groove _down)
SUBDIV = 5        # icosphere subdivisions (10,242 verts, 20,480 faces per ellipsoid)


def parse_obj_per_object(path):
    """Return [(name, V_Nx3, F_Mx3), ...] per `o` block in the OBJ."""
    verts = []
    groups = []
    cur_name = None
    cur_faces = []

    def flush():
        if cur_name is not None and cur_faces:
            groups.append((cur_name, list(cur_faces)))

    with open(path) as f:
        for line in f:
            t = line.split()
            if not t or t[0].startswith('#'):
                continue
            if t[0] == 'v':
                verts.append((float(t[1]), float(t[2]), float(t[3])))
            elif t[0] == 'o':
                flush()
                cur_name = ' '.join(t[1:]) or Path(path).stem
                cur_faces = []
            elif t[0] == 'f':
                idx = [int(s.split('/')[0]) - 1 for s in t[1:]]
                if len(idx) < 3:
                    continue
                for i in range(1, len(idx) - 1):
                    cur_faces.append((idx[0], idx[i], idx[i + 1]))
    flush()
    V_all = np.asarray(verts, dtype=np.float64)
    out = []
    for name, faces in groups:
        F = np.asarray(faces, dtype=np.int64)
        used = np.unique(F)
        remap = -np.ones(len(V_all), dtype=np.int64)
        remap[used] = np.arange(len(used))
        out.append((name, V_all[used], remap[F]))
    return out


def fit_ellipsoid(V, xz_margin=XZ_MARGIN, depth_floor=DEPTH_FLOOR):
    """Fit oriented ellipsoid to verts V (Nx3).
    Returns (verts, faces) of an icosphere scaled+oriented+translated.
    Axes: two in the best-fit plane, one perpendicular.
    Planimetric (XZ-world) margin then applied as a final post-scale.
    """
    centroid = V.mean(axis=0)
    rel = V - centroid
    # SVD: principal axes
    _, S, Vt = np.linalg.svd(rel, full_matrices=False)
    axes = Vt  # rows are principal directions, sorted by descending S
    # extents along each principal axis
    half_ext = np.zeros(3)
    for i in range(3):
        proj = rel @ axes[i]
        half_ext[i] = (proj.max() - proj.min()) / 2
    # 3rd axis (smallest variance) is the plane normal → enforce a min depth
    half_ext[2] = max(half_ext[2], depth_floor / 2)

    # Build icosphere, scale in axis-aligned local frame
    sph = trimesh.creation.icosphere(subdivisions=SUBDIV)
    vlocal = sph.vertices * half_ext  # scale each unit-sphere axis

    # Rotate from local axes -> world (R = axes^T  s.t. axes[0]_world = R @ [1,0,0])
    R = axes.T
    vworld = vlocal @ R.T

    # Planimetric XZ margin: scale world X and Z components from world centroid
    vworld[:, 0] *= (1.0 + xz_margin)
    vworld[:, 2] *= (1.0 + xz_margin)

    # Translate to centroid
    vworld = vworld + centroid
    return vworld, sph.faces


def write_obj(path, V, F, name="Ellipsoids"):
    with open(path, 'w') as f:
        f.write(f"# {name} - per-pad groove replacement ellipsoids\n")
        f.write(f"o {name}\n")
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in F:
            f.write("f " + " ".join(str(i + 1) for i in face) + "\n")


def main():
    out_dir = Path("pipeline_output")
    out_dir.mkdir(exist_ok=True)

    sources = {
        "central": "data/grooves/grooves_central.obj",
        "inner":   "data/grooves/grooves_inner.obj",
        "outer":   "data/grooves/grooves_outer.obj",
    }

    for ring, src in sources.items():
        p = Path(src)
        if not p.exists():
            print(f"SKIP missing: {src}")
            continue
        groups = parse_obj_per_object(src)
        print(f"\n{src}: {len(groups)} pad-groove(s)")
        all_v, all_f = [], []
        v_offset = 0
        for name, V, F in groups:
            ev, ef = fit_ellipsoid(V)
            all_v.append(ev)
            all_f.append(ef + v_offset)
            v_offset += len(ev)
            cx, cy, cz = V.mean(axis=0)
            half_dim = (V.max(axis=0) - V.min(axis=0)) / 2
            print(f"  {name:20s}  centroid=({cx:+7.1f},{cy:+7.1f},{cz:+7.1f})  "
                  f"bbox_half=({half_dim[0]:.1f},{half_dim[1]:.1f},{half_dim[2]:.1f})")
        if not all_v:
            continue
        V_out = np.vstack(all_v)
        F_out = np.vstack(all_f)
        obj_path = out_dir / f"groove_ellipsoids_{ring}.obj"
        stl_path = out_dir / f"groove_ellipsoids_{ring}.stl"
        write_obj(obj_path, V_out, F_out, name=f"GrooveEllipsoids_{ring}")
        trimesh.Trimesh(vertices=V_out, faces=F_out, process=False).export(stl_path)
        print(f"  Wrote {obj_path}  ({len(V_out)} verts, {len(F_out)} faces)")
        print(f"  Wrote {stl_path}")


if __name__ == "__main__":
    main()
