#!/usr/bin/env python3
"""Interactive 3D viewer for pan body voxel grids (.npz files).

Uses iy_top_map (playing surface height) and iy_base_top_cyl (base plate top)
to render the top surface and base boundary with proper interpretation.
"""

import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="View a pan body .npz voxel grid in 3D")
    parser.add_argument("npz_file", nargs="?",
                        default="data/pan_body_0.1mm_grid.npz",
                        help="Path to .npz grid file")
    parser.add_argument("--downsample", "-d", type=int, default=None,
                        help="Downsample factor (auto-chosen if omitted)")
    parser.add_argument("--mode", choices=["surface", "voxels", "pieces"],
                        default="surface",
                        help="surface: top surface mesh, voxels: solid point cloud, "
                             "pieces: colored by piece_xz assignment")
    parser.add_argument("--max-points", type=int, default=500_000,
                        help="Target max surface points (controls auto-downsample)")
    args = parser.parse_args()

    print(f"Loading {args.npz_file} ...")
    d = np.load(args.npz_file)

    print("Arrays in file:", list(d.keys()))
    for k in d:
        print(f"  {k}: shape={d[k].shape}, dtype={d[k].dtype}")

    x_range = d["x_range"]
    y_range = d["y_range"]
    z_range = d["z_range"]
    res = float(d["res"][0]) if "res" in d else 1.0
    iy_top_map = d["iy_top_map"]        # (nx, nz) — Y index of top surface per column
    iy_base_top = int(d["iy_base_top_cyl"][0])  # Y index where base plate ends

    print(f"\nResolution: {res}mm")
    print(f"Base plate top: y_index={iy_base_top}, "
          f"y={y_range[min(iy_base_top, len(y_range)-1)]:.1f}mm")
    print(f"Top map range: iy=[{iy_top_map.min()}, {iy_top_map.max()}], "
          f"y=[{y_range[max(0, iy_top_map.min())]:.1f}, "
          f"{y_range[min(iy_top_map.max(), len(y_range)-1)]:.1f}]mm")

    import plotly.graph_objects as go

    if args.mode == "surface":
        _show_surface(args, d, x_range, y_range, z_range, res,
                      iy_top_map, iy_base_top, go)
    elif args.mode == "pieces":
        _show_surface(args, d, x_range, y_range, z_range, res,
                      iy_top_map, iy_base_top, go, color_by_piece=True)
    else:
        _show_voxels(args, d, x_range, y_range, z_range, res, go)


def _show_surface(args, d, x_range, y_range, z_range, res,
                  iy_top_map, iy_base_top, go, color_by_piece=False):
    """Show the playing surface as a mesh using iy_top_map."""
    nx, nz = iy_top_map.shape
    piece_xz = d["piece_xz"]  # (nx, nz)

    # Downsample the 2D maps
    ds = args.downsample
    if ds is None:
        n_valid = np.count_nonzero(iy_top_map > 0)
        ds = max(1, int(np.ceil(np.sqrt(n_valid / args.max_points))))
        print(f"Valid surface points: {n_valid:,}, auto-downsample={ds}x")

    top = iy_top_map[::ds, ::ds]
    pxz = piece_xz[::ds, ::ds]
    xr = x_range[::ds]
    zr = z_range[::ds]
    snx, snz = top.shape

    # Mask: only columns with a valid surface (iy > 0 means there's geometry)
    valid = top > 0

    # Build X, Z grids and Y from the height map
    Xg, Zg = np.meshgrid(xr, zr, indexing="ij")
    # Clip indices to y_range bounds
    top_clipped = np.clip(top, 0, len(y_range) - 1)
    Yg = y_range[top_clipped]
    # Zero out invalid columns
    Yg[~valid] = np.nan

    print(f"Surface grid: {snx}x{snz}, {np.count_nonzero(valid):,} valid points")

    # Build triangulated mesh from the grid for proper surface rendering
    verts = []
    faces = []
    colors = []
    vert_idx = np.full((snx, snz), -1, dtype=int)

    # Collect valid vertices
    count = 0
    for i in range(snx):
        for j in range(snz):
            if valid[i, j]:
                verts.append([Xg[i, j], Zg[i, j], Yg[i, j]])
                if color_by_piece:
                    colors.append(float(pxz[i, j]))
                else:
                    colors.append(Yg[i, j])
                vert_idx[i, j] = count
                count += 1

    # Build triangle faces from adjacent valid quads
    for i in range(snx - 1):
        for j in range(snz - 1):
            v00 = vert_idx[i, j]
            v10 = vert_idx[i+1, j]
            v01 = vert_idx[i, j+1]
            v11 = vert_idx[i+1, j+1]
            if v00 >= 0 and v10 >= 0 and v01 >= 0 and v11 >= 0:
                faces.append([v00, v10, v11])
                faces.append([v00, v11, v01])

    verts = np.array(verts)
    faces = np.array(faces)
    colors = np.array(colors)
    print(f"Mesh: {len(verts):,} vertices, {len(faces):,} triangles")

    # Add base plate level as a reference ring
    traces = []

    # Main surface
    traces.append(go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=colors,
        colorscale="Turbo" if color_by_piece else "Viridis",
        colorbar=dict(title="Piece ID" if color_by_piece else "Y (mm)"),
        opacity=1.0,
        name="Playing surface",
    ))

    # Base plate reference circle
    base_y = y_range[min(iy_base_top, len(y_range) - 1)]
    theta = np.linspace(0, 2 * np.pi, 200)
    # Use the extent of valid surface to estimate drum radius
    valid_r = np.sqrt(Xg[valid]**2 + Zg[valid]**2)
    drum_r = np.max(valid_r)
    traces.append(go.Scatter3d(
        x=drum_r * np.cos(theta),
        y=drum_r * np.sin(theta),
        z=np.full_like(theta, base_y),
        mode="lines",
        line=dict(color="red", width=4),
        name=f"Base plate top (y={base_y:.1f}mm)",
    ))

    fig = go.Figure(data=traces)

    # Equal spacing in all dimensions
    x_ext = float(xr[-1] - xr[0])
    z_ext = float(zr[-1] - zr[0])
    y_ext = float(y_range[-1] - y_range[0])
    max_ext = max(x_ext, z_ext, y_ext)

    fig.update_layout(
        title=f"{args.npz_file} — {'pieces' if color_by_piece else 'surface'} "
              f"({len(verts):,} verts, ds={ds}x)",
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Z (mm)",
            zaxis_title="Y (mm)",
            aspectmode="manual",
            aspectratio=dict(
                x=x_ext / max_ext,
                y=z_ext / max_ext,
                z=y_ext / max_ext,
            ),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    suffix = "_pieces" if color_by_piece else "_surface"
    _save_and_open(fig, args.npz_file, suffix)


def _show_voxels(args, d, x_range, y_range, z_range, res, go):
    """Show solid voxels as a point cloud."""
    grid = d["body"]

    ds = args.downsample
    if ds is None:
        mid = grid.shape[0] // 2
        sample = grid[mid, :, :]
        frac = np.count_nonzero(sample) / max(sample.size, 1)
        est_solid = frac * grid.size
        ds = max(1, int(np.ceil((est_solid / args.max_points) ** (1/3))))
        print(f"Estimated ~{est_solid:,.0f} solid voxels, auto-downsample={ds}x")

    if ds > 1:
        print(f"Downsampling grid by {ds}x ...")
        grid = grid[::ds, ::ds, ::ds]
        x_range = x_range[::ds]
        y_range = y_range[::ds]
        z_range = z_range[::ds]
        print(f"Downsampled grid shape: {grid.shape}")

    print("Finding solid voxels ...")
    solid = np.argwhere(grid > 0)
    n_pts = len(solid)

    if n_pts > args.max_points:
        print(f"Subsampling {n_pts:,} -> {args.max_points:,} points ...")
        idx = np.random.default_rng(42).choice(n_pts, args.max_points, replace=False)
        idx.sort()
        solid = solid[idx]
        n_pts = len(solid)

    x = x_range[solid[:, 0]]
    y = y_range[solid[:, 1]]
    z = z_range[solid[:, 2]]

    print(f"Displaying {n_pts:,} voxel points ...")

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=z, z=y,
        mode="markers",
        marker=dict(
            size=max(1, min(3, 1000000 // max(n_pts, 1))),
            color=y, colorscale="Viridis",
            colorbar=dict(title="Y (mm)"),
            opacity=0.6,
        ),
    )])

    x_ext = float(x_range[-1] - x_range[0])
    z_ext = float(z_range[-1] - z_range[0])
    y_ext = float(y_range[-1] - y_range[0])
    max_ext = max(x_ext, z_ext, y_ext)

    fig.update_layout(
        title=f"{args.npz_file} — voxels ({n_pts:,} pts, ds={ds}x)",
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Z (mm)",
            zaxis_title="Y (mm)",
            aspectmode="manual",
            aspectratio=dict(
                x=x_ext / max_ext,
                y=z_ext / max_ext,
                z=y_ext / max_ext,
            ),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    _save_and_open(fig, args.npz_file, "_voxels")


def _save_and_open(fig, npz_file, suffix):
    out_html = npz_file.replace(".npz", f"{suffix}_view.html")
    fig.write_html(out_html)
    print(f"Saved to {out_html}")
    import webbrowser, os
    webbrowser.open("file://" + os.path.abspath(out_html))


if __name__ == "__main__":
    main()
