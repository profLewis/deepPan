#!/usr/bin/env python3
"""
Generate a combined OBJ with all pads + mount bases + outer sleeves
positioned in-place on the pan body, for visualization in Blender.

Coordinates match pan_body.obj.
"""

import numpy as np
from pathlib import Path
import json
import math

from generate_notepad import (
    parse_obj_file, extract_object_mesh, compute_surface_normal,
    compute_interior_centroid, compute_cylinder_surface_offset,
    transform_cylinder_to_normal, compute_leveling_rotation,
    find_boundary_edges, find_all_boundary_loops,
    NOTE_BY_INDEX, CM_TO_MM, PAN_SCALE, PAN_THICKNESS, check_and_scale_pad,
    MOUNT_INNER_DIAMETER, MOUNT_WALL_THICKNESS, MOUNT_THREAD_DEPTH, MOUNT_DEPTH,
    MIN_PAD_SIZE,
    CENTRAL_MOUNT_INNER_DIAMETER, CENTRAL_MOUNT_WALL_THICKNESS,
    CENTRAL_MOUNT_DEPTH, CENTRAL_MIN_PAD_SIZE,
)
from generate_mount_base import generate_cylinder as generate_large_mount_base
from generate_mount_base import generate_pcb_mount as generate_large_pcb_mount
from generate_outer_sleeve import generate_sleeve as generate_large_sleeve
from generate_central_mount import (
    generate_central_mount, generate_bounding_ellipse_prism, generate_pcb_prototype,
)


OBJ_PATH = "data/Tenor Pan only.obj"
OUTPUT_PATH = "data/notepads/assembly_view.obj"


def load_pan_centroid_offset():
    path = Path("data/pan_centroid_offset.json")
    if path.exists():
        with open(path) as f:
            return np.array(json.load(f)["centroid_offset_mm"])
    return np.zeros(3)


def to_body_coords(verts, pan_centroid_offset, R_level):
    """Transform from internal mm coords to pan_body coords (centered + leveled)."""
    return (R_level @ (verts - pan_centroid_offset).T).T


def main():
    print("Generating assembly view...")

    # Pre-compute transforms
    pan_centroid_offset = load_pan_centroid_offset()
    R_level = compute_leveling_rotation(OBJ_PATH)

    # Parse source OBJ once
    objects, all_vertices = parse_obj_file(OBJ_PATH)

    # Generate large mount (outer ring) and small mount (central/inner ring)
    print("Generating large mount base (outer ring)...")
    lg_base_v, lg_base_f = generate_large_mount_base()
    lg_pcb_v, lg_pcb_f = generate_large_pcb_mount()
    n_lg = len(lg_base_v)
    lg_mount_v = np.vstack([lg_base_v, lg_pcb_v])
    lg_mount_f = lg_base_f + [[i + n_lg for i in f] for f in lg_pcb_f]

    print("Generating large outer sleeve (outer ring)...")
    lg_sleeve_v, lg_sleeve_f = generate_large_sleeve()

    print("Generating central/inner push-cap mount...")
    pushcap_v, pushcap_f = generate_central_mount()

    print("Generating bounding ellipse prism + PCB prototype...")
    bc_v, bc_f, bc_sx, bc_sy, bc_zt, bc_zb = generate_bounding_ellipse_prism(tolerance=1.5)
    pcb_v, pcb_f = generate_pcb_prototype()

    # Per-ring mount parameters
    ring_params = {
        'outer': {
            'mount_v': lg_mount_v, 'mount_f': lg_mount_f,
            'sleeve_v': lg_sleeve_v, 'sleeve_f': lg_sleeve_f,
            'depth': MOUNT_DEPTH,
            'cyl_outer_r': MOUNT_INNER_DIAMETER / 2 + MOUNT_WALL_THICKNESS + MOUNT_THREAD_DEPTH,
            'min_pad_size': MIN_PAD_SIZE,
            'has_sleeve': True,
        },
        'central': {
            'mount_v': pushcap_v, 'mount_f': pushcap_f,
            'sleeve_v': None, 'sleeve_f': None,
            'depth': CENTRAL_MOUNT_DEPTH,
            'cyl_outer_r': CENTRAL_MOUNT_INNER_DIAMETER / 2 + CENTRAL_MOUNT_WALL_THICKNESS,
            'min_pad_size': CENTRAL_MIN_PAD_SIZE,
            'has_sleeve': False,
        },
        'inner': {
            'mount_v': pushcap_v, 'mount_f': pushcap_f,
            'sleeve_v': None, 'sleeve_f': None,
            'depth': CENTRAL_MOUNT_DEPTH,
            'cyl_outer_r': CENTRAL_MOUNT_INNER_DIAMETER / 2 + CENTRAL_MOUNT_WALL_THICKNESS,
            'min_pad_size': CENTRAL_MIN_PAD_SIZE,
            'has_sleeve': False,
        },
    }

    # For outer ring: compute a consistent angular orientation.
    # We want the wire notch (angle=0 in local coords) to point radially
    # outward from the pan center for all outer ring pads.
    # The pan center in the source coordinate system is at the pan_centroid_offset.
    pan_center_xz = pan_centroid_offset[[0, 2]]  # X, Z in original mm coords

    with open(OUTPUT_PATH, 'w') as out:
        out.write("# Assembly view: pads + mount bases + outer sleeves\n")
        out.write("# Coordinates match pan_body.obj\n\n")

        vert_offset = 0

        for note_index in sorted(NOTE_BY_INDEX.keys()):
            info = NOTE_BY_INDEX[note_index]
            ring = info['ring']
            pan_obj = info['pan_object']
            grove_obj = info['grove_object']

            print(f"  {note_index} ({info['note']}{info['octave']}) - {ring}...")

            rp = ring_params[ring]

            # Extract pad surface
            pan_verts, pan_faces = extract_object_mesh(objects, pan_obj, all_vertices)
            grove_verts, grove_faces = extract_object_mesh(objects, grove_obj, all_vertices)
            pan_verts, grove_verts, _, _ = check_and_scale_pad(
                pan_verts, grove_verts, min_size=rp['min_pad_size'])

            pan_normal = compute_surface_normal(pan_verts, pan_faces)
            _, pan_surface_centroid = compute_interior_centroid(
                pan_verts, pan_faces, pan_normal, PAN_THICKNESS, 0)

            # For outer ring, use boundary-loop centroid for consistent positioning
            if ring == 'outer':
                be = find_boundary_edges(pan_faces)
                loops = find_all_boundary_loops(be)
                if loops:
                    boundary_vi = max(loops, key=len)
                    boundary_center = pan_verts[boundary_vi].mean(axis=0)
                else:
                    boundary_center = (pan_verts.max(axis=0) + pan_verts.min(axis=0)) / 2
                diff = boundary_center - pan_surface_centroid
                pan_surface_centroid = boundary_center - np.dot(diff, pan_normal) * pan_normal

            # Compute surface offset for mount
            surface_offset = compute_cylinder_surface_offset(
                pan_verts, pan_surface_centroid, pan_normal, rp['cyl_outer_r'])
            mount_origin = pan_surface_centroid.copy()
            if surface_offset < 0:
                mount_origin = mount_origin + pan_normal * surface_offset

            # Compute rotation about the normal axis to orient mount hardware.
            # Outer ring: notch points radially outward from pan center.
            # Central/inner: plate aligns with pad's longest axis.
            if ring == 'outer':
                pad_xz = pan_surface_centroid[[0, 2]]
                radial_dir = pad_xz - pan_center_xz
                radial_len = np.linalg.norm(radial_dir)
                if radial_len > 1e-6:
                    radial_dir = radial_dir / radial_len
                else:
                    radial_dir = np.array([1.0, 0.0])
                radial_3d = np.array([radial_dir[0], 0.0, radial_dir[1]])
                target_tangent = radial_3d - np.dot(radial_3d, pan_normal) * pan_normal
                tlen = np.linalg.norm(target_tangent)
                target_tangent = target_tangent / tlen if tlen > 1e-6 else None
            else:
                # Central/inner: compute pad long axis via PCA on tangent plane
                n_hat = pan_normal / np.linalg.norm(pan_normal)
                if abs(n_hat[0]) < 0.9:
                    _xl = np.cross(n_hat, [1, 0, 0])
                else:
                    _xl = np.cross(n_hat, [0, 1, 0])
                _xl /= np.linalg.norm(_xl)
                _yl = np.cross(n_hat, _xl)
                _rel = pan_verts - pan_surface_centroid
                _lx, _ly = _rel @ _xl, _rel @ _yl
                _cov = np.cov(np.column_stack([_lx, _ly]).T)
                _vals, _vecs = np.linalg.eigh(_cov)
                _long_2d = _vecs[:, np.argmax(_vals)]
                target_tangent = _long_2d[0] * _xl + _long_2d[1] * _yl
                tlen = np.linalg.norm(target_tangent)
                target_tangent = target_tangent / tlen if tlen > 1e-6 else None

            def place_component(comp_verts, z_offset=0.0):
                """Transform component verts from local (Z-down) to pan_body coords."""
                v = comp_verts.copy()
                if z_offset != 0:
                    v[:, 2] += z_offset
                transformed = transform_cylinder_to_normal(v, mount_origin, pan_normal)

                if target_tangent is not None:
                    # Compute current local X-axis direction
                    # The standard transform maps local +X to some direction in tangent plane.
                    # We want it to map to target_tangent.
                    local_x = transform_cylinder_to_normal(
                        np.array([[1, 0, 0]]), np.zeros(3), pan_normal)[0]
                    local_x = local_x - np.dot(local_x, pan_normal) * pan_normal
                    local_x_len = np.linalg.norm(local_x)
                    if local_x_len > 1e-6:
                        local_x = local_x / local_x_len
                        # Angle between current local_x and desired target_tangent
                        cos_a = np.clip(np.dot(local_x, target_tangent), -1, 1)
                        sin_a = np.dot(np.cross(local_x, target_tangent), pan_normal)
                        angle = math.atan2(sin_a, cos_a)
                        # Rotate about the normal axis
                        c, s = math.cos(angle), math.sin(angle)
                        n = pan_normal / np.linalg.norm(pan_normal)
                        # Rodrigues rotation about n by angle
                        centered = transformed - mount_origin
                        rotated = (centered * c +
                                   np.cross(n, centered) * s +
                                   n * np.dot(centered, n).reshape(-1, 1) * (1 - c))
                        transformed = rotated + mount_origin

                return to_body_coords(transformed, pan_centroid_offset, R_level)

            # Load the pad's OBJ (already generated in pan_body coords)
            pad_obj_path = f"data/notepads/notepad_{note_index}.obj"
            pad_verts = []
            pad_faces_raw = []
            with open(pad_obj_path) as pf:
                for line in pf:
                    if line.startswith('v '):
                        pad_verts.append([float(x) for x in line.split()[1:4]])
                    elif line.startswith('f '):
                        face = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
                        pad_faces_raw.append(face)
            pad_verts = np.array(pad_verts)

            # Write pad
            out.write(f"o Pad_{note_index}\n")
            for v in pad_verts:
                out.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in pad_faces_raw:
                out.write(f"f {' '.join(str(i + 1 + vert_offset) for i in face)}\n")
            vert_offset += len(pad_verts)

            # Write mount base (ring-specific)
            z_off = -rp['depth']
            body_base_verts = place_component(rp['mount_v'], z_off)
            out.write(f"o MountBase_{note_index}\n")
            for v in body_base_verts:
                out.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in rp['mount_f']:
                out.write(f"f {' '.join(str(i + 1 + vert_offset) for i in face)}\n")
            vert_offset += len(body_base_verts)

            # Write sleeve (outer ring only — central/inner use push-cap)
            if rp['has_sleeve']:
                body_sleeve_verts = place_component(rp['sleeve_v'], z_off)
                out.write(f"o Sleeve_{note_index}\n")
                for v in body_sleeve_verts:
                    out.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                for face in rp['sleeve_f']:
                    out.write(f"f {' '.join(str(i + 1 + vert_offset) for i in face)}\n")
                vert_offset += len(body_sleeve_verts)

            # Write bounding ellipse + PCB prototype (central/inner only)
            if ring in ('central', 'inner'):
                body_bc_verts = place_component(bc_v, z_off)
                out.write(f"o BoundingEllipse_{note_index}\n")
                for v in body_bc_verts:
                    out.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                for face in bc_f:
                    out.write(f"f {' '.join(str(i + 1 + vert_offset) for i in face)}\n")
                vert_offset += len(body_bc_verts)

                body_pcb_verts = place_component(pcb_v, z_off)
                out.write(f"o PCBPrototype_{note_index}\n")
                for v in body_pcb_verts:
                    out.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                for face in pcb_f:
                    out.write(f"f {' '.join(str(i + 1 + vert_offset) for i in face)}\n")
                vert_offset += len(body_pcb_verts)

            out.write("\n")

    # Also write bounding ellipses to a separate file for use in pan body carving
    bc_output = "data/notepads/bounding_ellipses.obj"
    print(f"\nWriting bounding ellipses to {bc_output}...")
    with open(bc_output, 'w') as bcf:
        bcf.write("# Bounding ellipses for central/inner ring mechanisms\n")
        bcf.write("# Body coords (matches assembly_view.obj)\n\n")
        bc_vert_offset = 0
        # Re-iterate to place bounding cylinders
        for note_index in sorted(NOTE_BY_INDEX.keys()):
            info = NOTE_BY_INDEX[note_index]
            ring = info['ring']
            if ring not in ('central', 'inner'):
                continue

            pan_obj = info['pan_object']
            grove_obj = info['grove_object']
            rp = ring_params[ring]

            pan_verts_bc, pan_faces_bc = extract_object_mesh(objects, pan_obj, all_vertices)
            grove_verts_bc, grove_faces_bc = extract_object_mesh(objects, grove_obj, all_vertices)
            pan_verts_bc, grove_verts_bc, _, _ = check_and_scale_pad(
                pan_verts_bc, grove_verts_bc, min_size=rp['min_pad_size'])

            pan_normal_bc = compute_surface_normal(pan_verts_bc, pan_faces_bc)
            _, pan_sc_bc = compute_interior_centroid(
                pan_verts_bc, pan_faces_bc, pan_normal_bc, PAN_THICKNESS, 0)

            surface_offset_bc = compute_cylinder_surface_offset(
                pan_verts_bc, pan_sc_bc, pan_normal_bc, rp['cyl_outer_r'])
            mount_origin_bc = pan_sc_bc.copy()
            if surface_offset_bc < 0:
                mount_origin_bc = mount_origin_bc + pan_normal_bc * surface_offset_bc

            # Recompute target_tangent for central/inner
            n_hat_bc = pan_normal_bc / np.linalg.norm(pan_normal_bc)
            if abs(n_hat_bc[0]) < 0.9:
                xl_bc = np.cross(n_hat_bc, [1, 0, 0])
            else:
                xl_bc = np.cross(n_hat_bc, [0, 1, 0])
            xl_bc /= np.linalg.norm(xl_bc)
            yl_bc = np.cross(n_hat_bc, xl_bc)
            rel_bc = pan_verts_bc - pan_sc_bc
            lx_bc, ly_bc = rel_bc @ xl_bc, rel_bc @ yl_bc
            cov_bc = np.cov(np.column_stack([lx_bc, ly_bc]).T)
            vals_bc, vecs_bc = np.linalg.eigh(cov_bc)
            long_2d_bc = vecs_bc[:, np.argmax(vals_bc)]
            tt_bc = long_2d_bc[0] * xl_bc + long_2d_bc[1] * yl_bc
            tt_len_bc = np.linalg.norm(tt_bc)
            tt_bc = tt_bc / tt_len_bc if tt_len_bc > 1e-6 else None

            # Place bounding ellipse
            def place_bc(comp_verts):
                v = comp_verts.copy()
                z_off_bc = -rp['depth']
                v[:, 2] += z_off_bc
                transformed = transform_cylinder_to_normal(v, mount_origin_bc, pan_normal_bc)
                if tt_bc is not None:
                    local_x = transform_cylinder_to_normal(
                        np.array([[1, 0, 0]]), np.zeros(3), pan_normal_bc)[0]
                    local_x = local_x - np.dot(local_x, pan_normal_bc) * pan_normal_bc
                    lx_len = np.linalg.norm(local_x)
                    if lx_len > 1e-6:
                        local_x = local_x / lx_len
                        cos_a = np.clip(np.dot(local_x, tt_bc), -1, 1)
                        sin_a = np.dot(np.cross(local_x, tt_bc), pan_normal_bc)
                        angle = math.atan2(sin_a, cos_a)
                        c, s = math.cos(angle), math.sin(angle)
                        n = pan_normal_bc / np.linalg.norm(pan_normal_bc)
                        centered = transformed - mount_origin_bc
                        rotated = (centered * c +
                                   np.cross(n, centered) * s +
                                   n * np.dot(centered, n).reshape(-1, 1) * (1 - c))
                        transformed = rotated + mount_origin_bc
                return to_body_coords(transformed, pan_centroid_offset, R_level)

            body_bc = place_bc(bc_v)
            bcf.write(f"o BoundingEllipse_{note_index}\n")
            for v in body_bc:
                bcf.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in bc_f:
                bcf.write(f"f {' '.join(str(i + 1 + bc_vert_offset) for i in face)}\n")
            bc_vert_offset += len(body_bc)

    print(f"  {bc_vert_offset // len(bc_v)} bounding ellipses written")

    # Export combined STL
    stl_path = OUTPUT_PATH.replace('.obj', '.stl')
    print(f"\n  Exporting STL: {stl_path}...")
    import struct
    # Re-read the OBJ we just wrote to get all vertices/faces
    stl_verts, stl_faces = [], []
    with open(OUTPUT_PATH) as fin:
        for line in fin:
            if line.startswith('v '):
                p = line.split()
                stl_verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith('f '):
                indices = [int(tok.split('/')[0]) - 1 for tok in line.split()[1:]]
                # Triangulate
                for i in range(1, len(indices) - 1):
                    stl_faces.append([indices[0], indices[i], indices[i + 1]])
    stl_verts = np.array(stl_verts)
    with open(stl_path, 'wb') as sf:
        sf.write(b'Assembly view STL'.ljust(80, b'\0'))
        sf.write(struct.pack('<I', len(stl_faces)))
        for tri in stl_faces:
            v0, v1, v2 = stl_verts[tri[0]], stl_verts[tri[1]], stl_verts[tri[2]]
            n = np.cross(v1 - v0, v2 - v0)
            nl = np.linalg.norm(n)
            if nl > 0:
                n /= nl
            sf.write(struct.pack('<3f', *n))
            sf.write(struct.pack('<3f', *v0))
            sf.write(struct.pack('<3f', *v1))
            sf.write(struct.pack('<3f', *v2))
            sf.write(struct.pack('<H', 0))
    print(f"  {len(stl_faces)} triangles")

    print(f"\nAssembly view saved: {OUTPUT_PATH}")
    print(f"  STL: {stl_path}")
    print(f"  Includes: pads, mounts, sleeves, bounding ellipses, PCB prototypes")


if __name__ == "__main__":
    main()
