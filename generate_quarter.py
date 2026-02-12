#!/usr/bin/env python3
"""
Generate 3D-printable pan sections (sixths) for the outer ring.

Each sixth:
- Contains 2 outer-ring notes with sunken notepad pockets
- Keeps the original drum wall geometry (no subdivision or holes)
- Subdivides only the playing surface for clean holes
- Has structural support walls along the 2 straight (radial) cut edges
  with bolt holes, finger joints, and cable routing slots
- Has a curved inner arc support with studs and wire hole
- Drum wall vertical along +Z (pan axis along Y)

Usage:
    python generate_quarter.py              # Generate S0 (example)
    python generate_quarter.py S1           # Generate specific section
    python generate_quarter.py --all        # Generate all 6 sections
    python generate_quarter.py --orient-only  # Just export oriented pan
"""

import numpy as np
from pathlib import Path
import math
import json
import sys
import struct

from generate_sector import (
    extract_bowl_surface, load_notepad_properties,
    remove_faces_in_circle,
    SECTOR_THICKNESS, POCKET_DEPTH, MOUNT_CLEARANCE_HOLE,
)
from generate_notepad import (
    compute_vertex_normals, find_boundary_edges, find_all_boundary_loops,
    PAN_THICKNESS, BOSS_OUTER_DIAMETER,
)

# ============================================================
# Constants
# ============================================================

BOSS_THROUGH_HOLE = 12.0        # mm, clearance for 10mm notepad boss
WALL_THICKNESS = 6.0            # mm, structural wall thickness
WALL_BOLT_SPACING = 50.0        # mm, bolt hole spacing along walls
WALL_BOLT_DIAMETER = 4.3        # mm, M4 clearance
INNER_CUTOFF_RADIUS = 145.0     # mm, remove bowl faces below this radius
WALL_INNER_RADIUS = 145.0       # mm, radial walls start at outer ring inner edge
WALL_PROFILE_SAMPLES = 40       # radial samples for wall top profile
SUBDIVIDE_ROUNDS = 3            # subdivision rounds on playing surface (for pocket resolution)
WALL_CABLE_HOLE_DIAMETER = 30.0 # mm, cable routing slot in each wall
FINGER_TAB_WIDTH = 12.0         # mm, finger joint tab height (vertical)
FINGER_TAB_DEPTH = 5.0          # mm, how far tabs protrude into mating section
N_FINGER_TABS = 3               # number of finger joint tabs per wall
DRUM_WALL_RADIUS = 225.0        # mm, faces beyond this with steep normals = drum wall
DRUM_WALL_NORMAL_Y = 0.5        # drum wall faces have |normal_y| below this

# Inner arc support constants
INNER_ARC_THICKNESS = 6.0       # mm, inner arc wall thickness
INNER_ARC_STUD_DIAMETER = 6.0   # mm, studs for attaching to central drum ring
INNER_ARC_STUD_HEIGHT = 5.0     # mm, stud protrusion
INNER_ARC_N_STUDS = 3           # studs per arc
INNER_ARC_WIRE_HOLE_WIDTH = 20.0  # mm, wire routing opening in arc

# Section definitions: (start_angle, end_angle, [note_names])
# 6 sections of 60° each, cuts placed in gaps between outer-ring notes
SECTIONS = {
    'S0': (15.0,   75.0,  ['O2', 'O1']),
    'S1': (75.0,  135.0,  ['O0', 'O11']),
    'S2': (135.0, 195.0,  ['O10', 'O9']),
    'S3': (195.0, 255.0,  ['O8', 'O7']),
    'S4': (255.0, 315.0,  ['O6', 'O5']),
    'S5': (315.0, 375.0,  ['O4', 'O3']),  # 375 = 15 + 360
}


# ============================================================
# Pan Orientation
# ============================================================

def compute_pan_rotation(bowl_verts, bowl_faces, face_material):
    """
    Compute rotation to orient the pan with drum wall vertical along +Z.
    Returns rotation matrix R such that: oriented = bowl_verts @ R.T
    """
    groves_vi = set()
    for fi, face in enumerate(bowl_faces):
        if face_material[fi] == 'Groves':
            for vi in face:
                groves_vi.add(vi)
    groves_verts = bowl_verts[sorted(groves_vi)]
    print(f"  Fitting plane to {len(groves_verts)} Groves vertices...")

    centered = groves_verts - groves_verts.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    drum_axis = Vt[2]
    if drum_axis[1] < 0:
        drum_axis = -drum_axis
    print(f"  Drum axis (original): ({drum_axis[0]:.4f}, {drum_axis[1]:.4f}, {drum_axis[2]:.4f})")

    z_axis = np.array([0.0, 0.0, 1.0])
    if np.allclose(drum_axis, z_axis, atol=1e-3):
        R1 = np.eye(3)
    else:
        v = np.cross(drum_axis, z_axis)
        s = np.linalg.norm(v)
        c = np.dot(drum_axis, z_axis)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R1 = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)

    R_extra = np.array([
        [1,  0,  0],
        [0,  0,  1],
        [0, -1,  0],
    ], dtype=float)

    R = R_extra @ R1
    return R


def apply_rotation_to_props(R, note_props):
    """Apply rotation matrix R to all notepad properties (in-place)."""
    for idx in note_props:
        note_props[idx]['centroid_pan'] = R @ note_props[idx]['centroid_pan']
        note_props[idx]['normal'] = R @ note_props[idx]['normal']
        note_props[idx]['normal'] /= np.linalg.norm(note_props[idx]['normal'])
        note_props[idx]['boss_positions_pan'] = [
            R @ bp for bp in note_props[idx]['boss_positions_pan']
        ]


# ============================================================
# Mesh Subdivision
# ============================================================

def subdivide_mesh(vertices, faces, *face_attrs_lists):
    """
    Midpoint subdivision: split each triangle into 4 by adding edge midpoints.
    Returns (new_vertices, new_faces, *new_attrs_lists).
    """
    edge_mids = {}
    new_verts = list(vertices)

    def get_mid(v1, v2):
        key = (min(v1, v2), max(v1, v2))
        if key not in edge_mids:
            edge_mids[key] = len(new_verts)
            new_verts.append((vertices[v1] + vertices[v2]) / 2.0)
        return edge_mids[key]

    new_faces = []
    new_attrs = [[] for _ in face_attrs_lists]

    for fi, face in enumerate(faces):
        attrs = [lst[fi] for lst in face_attrs_lists]
        triangles = []
        if len(face) == 3:
            triangles = [face]
        else:
            for i in range(1, len(face) - 1):
                triangles.append([face[0], face[i], face[i + 1]])

        for tri in triangles:
            a, b, c = tri
            d, e, f = get_mid(a, b), get_mid(b, c), get_mid(a, c)
            for nf in [[a, d, f], [d, b, e], [f, e, c], [d, e, f]]:
                new_faces.append(nf)
                for j in range(len(new_attrs)):
                    new_attrs[j].append(attrs[j])

    result = [np.array(new_verts), new_faces]
    result.extend(new_attrs)
    return tuple(result)


# ============================================================
# Angular helpers
# ============================================================

def normalize_angle(a):
    """Normalize angle to (-180, 180] range."""
    while a <= -180:
        a += 360
    while a > 180:
        a -= 360
    return a


def angle_in_sector(ang, start, end):
    """Check if angle is within sector from start to end (counter-clockwise)."""
    a = normalize_angle(ang - start)
    span = normalize_angle(end - start)
    if span <= 0:
        span += 360
    if a < 0:
        a += 360
    return a < span


# ============================================================
# Cutting & Reindexing
# ============================================================

def cut_quarter(vertices, faces, angle_start, angle_end, *face_attrs_lists):
    """Extract faces within an angular sector in the X-Z plane."""
    out_faces = []
    out_attrs = [[] for _ in face_attrs_lists]

    for fi, face in enumerate(faces):
        cx = np.mean([vertices[vi][0] for vi in face])
        cz = np.mean([vertices[vi][2] for vi in face])
        ang = math.degrees(math.atan2(cz, cx))

        if angle_in_sector(ang, angle_start, angle_end):
            out_faces.append(face)
            for j in range(len(out_attrs)):
                out_attrs[j].append(face_attrs_lists[j][fi])

    result = [out_faces]
    result.extend(out_attrs)
    return tuple(result)


def reindex_mesh(vertices, faces):
    """Re-index mesh to remove unused vertices."""
    used = set()
    for face in faces:
        for vi in face:
            used.add(vi)
    sorted_used = sorted(used)
    remap = {old: new for new, old in enumerate(sorted_used)}
    new_verts = vertices[sorted_used]
    new_faces = [[remap[vi] for vi in face] for face in faces]
    return new_verts, new_faces


# ============================================================
# Inner area removal
# ============================================================

def remove_inner_faces(vertices, faces, inner_radius, *face_attrs_lists):
    """Remove faces whose centroid is inside inner_radius in the X-Z plane."""
    out_faces = []
    out_attrs = [[] for _ in face_attrs_lists]

    for fi, face in enumerate(faces):
        cx = np.mean([vertices[vi][0] for vi in face])
        cz = np.mean([vertices[vi][2] for vi in face])
        r = math.sqrt(cx * cx + cz * cz)
        if r >= inner_radius:
            out_faces.append(face)
            for j in range(len(out_attrs)):
                out_attrs[j].append(face_attrs_lists[j][fi])

    result = [out_faces]
    result.extend(out_attrs)
    return tuple(result)


# ============================================================
# Drum wall classification
# ============================================================

def compute_face_normals(vertices, faces):
    """Compute per-face normals."""
    normals = []
    for face in faces:
        if len(face) < 3:
            normals.append(np.array([0.0, 1.0, 0.0]))
            continue
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        n = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(n)
        normals.append(n / norm if norm > 0 else np.array([0.0, 1.0, 0.0]))
    return normals


def classify_drum_wall(vertices, faces):
    """
    Classify faces as drum wall vs playing surface.

    Drum wall: faces at the outer perimeter with steep (nearly vertical) normals.
    Playing surface: everything else (the concave bowl area with notes).

    Returns per-face boolean list (True = drum wall).
    """
    normals = compute_face_normals(vertices, faces)
    is_drum_wall = []
    n_dw = 0
    for fi, face in enumerate(faces):
        fc = vertices[face].mean(axis=0)
        r = math.sqrt(fc[0]**2 + fc[2]**2)
        ny = abs(normals[fi][1])
        dw = r > DRUM_WALL_RADIUS and ny < DRUM_WALL_NORMAL_Y
        is_drum_wall.append(dw)
        if dw:
            n_dw += 1
    print(f"  Drum wall: {n_dw} faces, Playing surface: {len(faces) - n_dw} faces")
    return is_drum_wall


# ============================================================
# Thickening
# ============================================================

def thicken_with_pocket(vertices, faces, is_pocket_face, thickness, pocket_depth):
    """
    Thicken surface mesh into a solid shell with differential pocket depth.
    Pocket faces stay at surface level. Rim faces are raised by pocket_depth.
    """
    normals = compute_vertex_normals(vertices, faces)
    n = len(vertices)

    v_pocket_count = np.zeros(n, dtype=int)
    v_face_count = np.zeros(n, dtype=int)
    for fi, face in enumerate(faces):
        for vi in face:
            v_face_count[vi] += 1
            if is_pocket_face[fi]:
                v_pocket_count[vi] += 1
    v_is_pocket = v_pocket_count == v_face_count

    top = np.copy(vertices)
    for i in range(n):
        if not v_is_pocket[i]:
            top[i] += normals[i] * pocket_depth

    bot = vertices - normals * thickness

    all_v = np.vstack([top, bot])
    top_f = [list(f) for f in faces]
    bot_f = [[vi + n for vi in reversed(f)] for f in faces]

    be = find_boundary_edges(faces)
    loops = find_all_boundary_loops(be)
    side_f = []
    for loop in loops:
        for i in range(len(loop)):
            v1, v2 = loop[i], loop[(i + 1) % len(loop)]
            side_f.append([v1, v2, v2 + n, v1 + n])

    return all_v, top_f + bot_f + side_f


# ============================================================
# Support Wall Construction (profile-following)
# ============================================================

def _inward_perp(cut_angle_deg, quarter_mid_angle_deg):
    """Compute the inward perpendicular direction for a cut wall."""
    cut_rad = math.radians(cut_angle_deg)
    mid_rad = math.radians(quarter_mid_angle_deg)
    cut_dx = math.cos(cut_rad)
    cut_dz = math.sin(cut_rad)
    perp_x = -math.sin(cut_rad)
    perp_z = math.cos(cut_rad)
    mid_dx = math.cos(mid_rad)
    mid_dz = math.sin(mid_rad)
    cross = cut_dx * mid_dz - cut_dz * mid_dx
    if cross < 0:
        perp_x, perp_z = -perp_x, -perp_z
    return cut_dx, cut_dz, perp_x, perp_z


def sample_surface_profile(bowl_verts, cut_angle_deg, r_min, r_max,
                            n_samples=WALL_PROFILE_SAMPLES):
    """Sample Y height of the bowl surface along a radial line at cut_angle_deg."""
    radii = np.sqrt(bowl_verts[:, 0]**2 + bowl_verts[:, 2]**2)
    angles = np.degrees(np.arctan2(bowl_verts[:, 2], bowl_verts[:, 0]))

    angle_diff = np.abs(angles - cut_angle_deg)
    angle_diff = np.minimum(angle_diff, 360.0 - angle_diff)
    near_mask = angle_diff < 5.0

    step = (r_max - r_min) / n_samples
    profile = []
    for i in range(n_samples + 1):
        r = r_min + step * i
        r_mask = near_mask & (np.abs(radii - r) < step * 1.5)
        if r_mask.any():
            y_top = bowl_verts[r_mask, 1].max()
        else:
            y_top = profile[-1][1] if profile else 0.0
        profile.append((r, y_top))

    return profile


def build_profiled_wall(cut_angle_deg, quarter_mid_angle_deg,
                        profile, base_y, thickness):
    """
    Build a support wall whose top edge follows the drum surface profile.
    Returns (wall_verts, wall_faces).
    """
    cut_dx, cut_dz, perp_x, perp_z = _inward_perp(
        cut_angle_deg, quarter_mid_angle_deg)

    n = len(profile)
    verts = []

    for r, y_top in profile:
        verts.append([r * cut_dx, y_top, r * cut_dz])
        verts.append([r * cut_dx, base_y, r * cut_dz])

    inner_off = n * 2
    for r, y_top in profile:
        verts.append([r * cut_dx + thickness * perp_x, y_top,
                      r * cut_dz + thickness * perp_z])
        verts.append([r * cut_dx + thickness * perp_x, base_y,
                      r * cut_dz + thickness * perp_z])

    faces = []
    for i in range(n - 1):
        ot0 = i * 2
        ob0 = i * 2 + 1
        ot1 = (i + 1) * 2
        ob1 = (i + 1) * 2 + 1
        it0 = inner_off + i * 2
        ib0 = inner_off + i * 2 + 1
        it1 = inner_off + (i + 1) * 2
        ib1 = inner_off + (i + 1) * 2 + 1

        faces.append([ot0, ob0, ob1, ot1])
        faces.append([it1, ib1, ib0, it0])
        faces.append([ot0, ot1, it1, it0])
        faces.append([ob1, ob0, ib0, ib1])

    faces.append([0, inner_off, inner_off + 1, 1])
    last_o = (n - 1) * 2
    last_i = inner_off + (n - 1) * 2
    faces.append([last_o, last_o + 1, last_i + 1, last_i])

    return np.array(verts, dtype=float), faces


def build_wall_with_hole(cut_angle_deg, quarter_mid_angle_deg,
                         profile, base_y, thickness,
                         hole_frac=0.70, hole_diameter=WALL_CABLE_HOLE_DIAMETER):
    """
    Build support wall with a cable routing slot cut through it.

    The wall is split into two radial segments with a gap between them.
    The gap creates a rectangular slot for passing cables.

    Returns (wall_verts, wall_faces).
    """
    r_min = profile[0][0]
    r_max = profile[-1][0]
    wall_len = r_max - r_min
    hole_r = hole_diameter / 2.0

    # Place hole at hole_frac along the wall (toward outer end)
    hole_r_pos = r_min + wall_len * hole_frac

    # Check if wall is long enough for a hole
    if wall_len < hole_diameter * 2.5:
        return build_profiled_wall(cut_angle_deg, quarter_mid_angle_deg,
                                   profile, base_y, thickness)

    # Check wall height at hole position
    prof_r = np.array([p[0] for p in profile])
    prof_y = np.array([p[1] for p in profile])
    y_top_at_hole = float(np.interp(hole_r_pos, prof_r, prof_y))
    wall_height = y_top_at_hole - base_y
    if wall_height < hole_diameter * 1.3:
        return build_profiled_wall(cut_angle_deg, quarter_mid_angle_deg,
                                   profile, base_y, thickness)

    # Split profile into inner and outer segments
    inner_profile = [(r, y) for r, y in profile if r <= hole_r_pos - hole_r]
    outer_profile = [(r, y) for r, y in profile if r >= hole_r_pos + hole_r]

    # Ensure at least 2 points per segment
    if len(inner_profile) < 2 or len(outer_profile) < 2:
        return build_profiled_wall(cut_angle_deg, quarter_mid_angle_deg,
                                   profile, base_y, thickness)

    # Add boundary points at exact hole edges
    y_inner_edge = float(np.interp(hole_r_pos - hole_r, prof_r, prof_y))
    y_outer_edge = float(np.interp(hole_r_pos + hole_r, prof_r, prof_y))
    inner_profile.append((hole_r_pos - hole_r, y_inner_edge))
    outer_profile.insert(0, (hole_r_pos + hole_r, y_outer_edge))

    all_verts = np.zeros((0, 3))
    all_faces = []

    # Build inner wall segment
    wv, wf = build_profiled_wall(cut_angle_deg, quarter_mid_angle_deg,
                                  inner_profile, base_y, thickness)
    all_verts = wv
    all_faces = list(wf)

    # Build outer wall segment
    wv, wf = build_profiled_wall(cut_angle_deg, quarter_mid_angle_deg,
                                  outer_profile, base_y, thickness)
    n_base = len(all_verts)
    all_verts = np.vstack([all_verts, wv])
    all_faces += [[vi + n_base for vi in f] for f in wf]

    return all_verts, all_faces


def add_bolt_features(cut_angle_deg, quarter_mid_angle_deg,
                      profile, base_y, spacing, diameter, thickness):
    """Create cylindrical bolt hole markers through the profiled wall."""
    cut_dx, cut_dz, perp_x, perp_z = _inward_perp(
        cut_angle_deg, quarter_mid_angle_deg)

    r_min = profile[0][0]
    r_max = profile[-1][0]
    wall_len = r_max - r_min
    if wall_len < spacing:
        return np.zeros((0, 3)), []

    n_bolts = max(1, int(wall_len / spacing))
    n_segs = 16
    ring_r = diameter / 2

    prof_r = np.array([p[0] for p in profile])
    prof_y = np.array([p[1] for p in profile])

    bolt_verts = []
    bolt_faces = []

    for bi in range(n_bolts):
        t = (bi + 0.5) / n_bolts
        r = r_min + t * wall_len
        y_top = float(np.interp(r, prof_r, prof_y))
        mid_y = (y_top + base_y) / 2.0
        cx, cz = r * cut_dx, r * cut_dz

        front_start = len(bolt_verts)
        for si in range(n_segs):
            a = 2 * math.pi * si / n_segs
            dy = ring_r * math.cos(a)
            dr = ring_r * math.sin(a)
            bolt_verts.append([cx + dr * cut_dx, mid_y + dy, cz + dr * cut_dz])

        back_start = len(bolt_verts)
        for si in range(n_segs):
            a = 2 * math.pi * si / n_segs
            dy = ring_r * math.cos(a)
            dr = ring_r * math.sin(a)
            bolt_verts.append([
                cx + thickness * perp_x + dr * cut_dx,
                mid_y + dy,
                cz + thickness * perp_z + dr * cut_dz,
            ])

        bolt_faces.append(list(range(front_start, front_start + n_segs)))
        bolt_faces.append(list(range(back_start + n_segs - 1, back_start - 1, -1)))
        for si in range(n_segs):
            s1 = (si + 1) % n_segs
            bolt_faces.append([
                front_start + si, front_start + s1,
                back_start + s1, back_start + si,
            ])

    if not bolt_verts:
        return np.zeros((0, 3)), []
    return np.array(bolt_verts), bolt_faces


# ============================================================
# Finger Joint Construction
# ============================================================

def build_finger_joints(cut_angle_deg, quarter_mid_angle_deg,
                        profile, base_y, wall_thickness,
                        n_tabs=N_FINGER_TABS, tab_width=FINGER_TAB_WIDTH,
                        tab_depth=FINGER_TAB_DEPTH):
    """
    Build interlocking finger joint tabs on the mating face of a support wall.
    Even-indexed segments get protruding tabs; adjacent sections have the
    complementary pattern.
    """
    cut_dx, cut_dz, perp_x, perp_z = _inward_perp(
        cut_angle_deg, quarter_mid_angle_deg)

    r_min = profile[0][0]
    r_max = profile[-1][0]
    wall_len = r_max - r_min

    prof_r = np.array([p[0] for p in profile])
    prof_y = np.array([p[1] for p in profile])

    n_slots = 2 * n_tabs + 1
    slot_len = wall_len / n_slots

    tab_verts = []
    tab_faces = []

    for i in range(n_slots):
        if i % 2 != 0:
            continue

        r_start = r_min + i * slot_len
        r_end = r_start + slot_len
        y_top_s = float(np.interp(r_start, prof_r, prof_y))
        y_top_e = float(np.interp(r_end, prof_r, prof_y))

        out_x = -perp_x * tab_depth
        out_z = -perp_z * tab_depth

        base = len(tab_verts)
        mid_y_s = (y_top_s + base_y) / 2.0
        mid_y_e = (y_top_e + base_y) / 2.0
        half_h = min(tab_width / 2, (y_top_s - base_y) * 0.3,
                     (y_top_e - base_y) * 0.3)

        v = [
            [r_start * cut_dx, mid_y_s + half_h, r_start * cut_dz],
            [r_start * cut_dx, mid_y_s - half_h, r_start * cut_dz],
            [r_end * cut_dx, mid_y_e - half_h, r_end * cut_dz],
            [r_end * cut_dx, mid_y_e + half_h, r_end * cut_dz],
            [r_start * cut_dx + out_x, mid_y_s + half_h, r_start * cut_dz + out_z],
            [r_start * cut_dx + out_x, mid_y_s - half_h, r_start * cut_dz + out_z],
            [r_end * cut_dx + out_x, mid_y_e - half_h, r_end * cut_dz + out_z],
            [r_end * cut_dx + out_x, mid_y_e + half_h, r_end * cut_dz + out_z],
        ]

        tab_verts.extend(v)

        b = base
        tab_faces.extend([
            [b+4, b+5, b+6, b+7],
            [b+0, b+3, b+7, b+4],
            [b+1, b+5, b+6, b+2],
            [b+0, b+4, b+5, b+1],
            [b+3, b+2, b+6, b+7],
        ])

    if not tab_verts:
        return np.zeros((0, 3)), []
    return np.array(tab_verts, dtype=float), tab_faces


# ============================================================
# Inner Arc Support
# ============================================================

def build_inner_arc_support(angle_start_deg, angle_end_deg, inner_radius,
                            y_top, base_y, thickness=INNER_ARC_THICKNESS,
                            n_segments=12):
    """
    Build curved support wall along the inner edge of a section.

    The arc is at R = inner_radius, spanning the section's angular range.
    Includes studs on the inner face (for attaching to central drum ring)
    and a wire routing gap in the middle.

    Returns (arc_verts, arc_faces).
    """
    # Handle wraparound
    arc_end = angle_end_deg
    if arc_end < angle_start_deg:
        arc_end += 360.0

    arc_span = arc_end - angle_start_deg
    # Gap for wire routing in the middle (angular extent)
    wire_gap_angle = math.degrees(INNER_ARC_WIRE_HOLE_WIDTH / inner_radius)
    gap_start_angle = angle_start_deg + (arc_span - wire_gap_angle) / 2
    gap_end_angle = gap_start_angle + wire_gap_angle

    all_verts = []
    all_faces = []

    # Build arc in two segments (with gap for wire hole)
    segments_list = [
        (angle_start_deg, gap_start_angle),
        (gap_end_angle, arc_end),
    ]

    for seg_start, seg_end in segments_list:
        seg_span = seg_end - seg_start
        if seg_span < 1.0:
            continue

        n_seg = max(3, int(n_segments * seg_span / arc_span))
        base_idx = len(all_verts)

        outer_r = inner_radius
        inner_r = inner_radius - thickness

        for i in range(n_seg + 1):
            t = i / n_seg
            a = math.radians(seg_start + t * seg_span)
            dx, dz = math.cos(a), math.sin(a)

            all_verts.append([outer_r * dx, y_top, outer_r * dz])
            all_verts.append([outer_r * dx, base_y, outer_r * dz])
            all_verts.append([inner_r * dx, y_top, inner_r * dz])
            all_verts.append([inner_r * dx, base_y, inner_r * dz])

        for i in range(n_seg):
            c = base_idx + i * 4
            nx = base_idx + (i + 1) * 4

            all_faces.append([c, c + 1, nx + 1, nx])         # outer
            all_faces.append([nx + 2, nx + 3, c + 3, c + 2]) # inner (reversed)
            all_faces.append([c, nx, nx + 2, c + 2])          # top
            all_faces.append([c + 1, c + 3, nx + 3, nx + 1])  # bottom

        # End caps
        all_faces.append([base_idx, base_idx + 2, base_idx + 3, base_idx + 1])
        last = base_idx + n_seg * 4
        all_faces.append([last, last + 1, last + 3, last + 2])

    # Add studs on inner face for connection to central drum ring
    stud_r = INNER_ARC_STUD_DIAMETER / 2.0
    n_stud_segs = 12

    for si in range(INNER_ARC_N_STUDS):
        t = (si + 0.5) / INNER_ARC_N_STUDS
        stud_angle = math.radians(angle_start_deg + t * arc_span)
        dx, dz = math.cos(stud_angle), math.sin(stud_angle)

        # Skip studs in the wire gap region
        stud_angle_deg = angle_start_deg + t * arc_span
        if gap_start_angle <= stud_angle_deg <= gap_end_angle:
            continue

        stud_cx = (inner_radius - thickness) * dx
        stud_cz = (inner_radius - thickness) * dz
        stud_cy = (y_top + base_y) / 2.0

        # Stud protrudes inward (toward center)
        stud_dir_x = -dx
        stud_dir_z = -dz

        base_idx = len(all_verts)

        # Stud cross-section axes: vertical (0,1,0) and tangential (-dz,0,dx)
        tan_x = -dz
        tan_z = dx

        # Base ring (on inner wall surface)
        for j in range(n_stud_segs):
            a = 2 * math.pi * j / n_stud_segs
            dy = stud_r * math.cos(a)
            dt = stud_r * math.sin(a)
            all_verts.append([
                stud_cx + dt * tan_x,
                stud_cy + dy,
                stud_cz + dt * tan_z,
            ])

        # Tip ring (protruding inward)
        for j in range(n_stud_segs):
            a = 2 * math.pi * j / n_stud_segs
            dy = stud_r * math.cos(a)
            dt = stud_r * math.sin(a)
            all_verts.append([
                stud_cx + stud_dir_x * INNER_ARC_STUD_HEIGHT + dt * tan_x,
                stud_cy + dy,
                stud_cz + stud_dir_z * INNER_ARC_STUD_HEIGHT + dt * tan_z,
            ])

        # Tip center
        tip_center = len(all_verts)
        all_verts.append([
            stud_cx + stud_dir_x * INNER_ARC_STUD_HEIGHT,
            stud_cy,
            stud_cz + stud_dir_z * INNER_ARC_STUD_HEIGHT,
        ])

        # Side wall
        for j in range(n_stud_segs):
            j_next = (j + 1) % n_stud_segs
            all_faces.append([
                base_idx + j, base_idx + j_next,
                base_idx + n_stud_segs + j_next,
                base_idx + n_stud_segs + j,
            ])

        # Tip cap (fan)
        for j in range(n_stud_segs):
            j_next = (j + 1) % n_stud_segs
            all_faces.append([
                tip_center,
                base_idx + n_stud_segs + j,
                base_idx + n_stud_segs + j_next,
            ])

    if not all_verts:
        return np.zeros((0, 3)), []
    return np.array(all_verts, dtype=float), all_faces


# ============================================================
# Output
# ============================================================

def write_quarter_stl(filepath, vertices, faces, name="Quarter"):
    """Write mesh to binary STL."""
    def face_normal(v0, v1, v2):
        e1 = v1 - v0
        e2 = v2 - v0
        n = np.cross(e1, e2)
        norm = np.linalg.norm(n)
        return n / norm if norm > 0 else n

    triangles = []
    for face in faces:
        if len(face) == 3:
            triangles.append(face)
        elif len(face) == 4:
            triangles.append([face[0], face[1], face[2]])
            triangles.append([face[0], face[2], face[3]])
        else:
            for i in range(1, len(face) - 1):
                triangles.append([face[0], face[i], face[i + 1]])

    with open(filepath, 'wb') as f:
        header = f"STL {name}".encode('ascii')[:80].ljust(80, b'\0')
        f.write(header)
        f.write(struct.pack('<I', len(triangles)))
        for tri in triangles:
            v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            n = face_normal(v0, v1, v2)
            f.write(struct.pack('<3f', *n))
            f.write(struct.pack('<3f', *v0))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<H', 0))

    print(f"  Saved: {filepath}")


def write_quarter_obj(filepath, vertices, faces, name="Quarter"):
    """Write mesh to OBJ."""
    with open(filepath, 'w') as f:
        f.write(f"# {name} - Pan Section\n")
        f.write(f"# Generated by generate_quarter.py\n# Units: mm\n\n")
        f.write(f"o {name}\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")
        for face in faces:
            f.write("f " + " ".join(str(idx + 1) for idx in face) + "\n")
    print(f"  Saved: {filepath}")


# ============================================================
# Prepare shells (playing surface + drum wall separately)
# ============================================================

def prepare_shells(bowl_verts, bowl_faces, face_material, note_props):
    """
    Process the bowl into two solid shells:

    1. Playing surface: subdivided, holes punched, thickened with pockets
    2. Drum wall: original geometry, thickened uniformly (no modification)

    Returns (ps_shell_v, ps_shell_f, dw_shell_v, dw_shell_f).
    """
    # Classify drum wall vs playing surface
    print(f"  Classifying drum wall vs playing surface...")
    is_dw = classify_drum_wall(bowl_verts, bowl_faces)

    # Split into separate meshes
    ps_faces = [f for f, dw in zip(bowl_faces, is_dw) if not dw]
    ps_mats = [m for m, dw in zip(face_material, is_dw) if not dw]
    dw_faces = [f for f, dw in zip(bowl_faces, is_dw) if dw]

    print(f"  Playing surface: {len(ps_faces)} faces")
    print(f"  Drum wall: {len(dw_faces)} faces (kept original)")

    # --- Playing surface pipeline ---
    ps_verts, ps_faces = reindex_mesh(bowl_verts, ps_faces)

    # Subdivide playing surface only
    for rnd in range(SUBDIVIDE_ROUNDS):
        print(f"  Subdivide playing surface round {rnd+1}/{SUBDIVIDE_ROUNDS}: "
              f"{len(ps_verts)} verts, {len(ps_faces)} faces ...")
        ps_verts, ps_faces, ps_mats = subdivide_mesh(ps_verts, ps_faces, ps_mats)
    print(f"  After subdivision: {len(ps_verts)} verts, {len(ps_faces)} faces")

    # Remove inner area from playing surface
    print(f"  Removing inner area (R < {INNER_CUTOFF_RADIUS}mm)...")
    ps_faces, ps_mats = remove_inner_faces(
        ps_verts, ps_faces, INNER_CUTOFF_RADIUS, ps_mats)
    print(f"  After inner cutout: {len(ps_faces)} faces")
    ps_verts, ps_faces = reindex_mesh(ps_verts, ps_faces)

    # Classify pocket faces for all outer-ring notes
    print(f"  Classifying pocket faces...")
    section_notes = set()
    for _, _, notes in SECTIONS.values():
        section_notes.update(notes)

    centroid_keys = list(note_props.keys())
    centroid_arr = np.array([note_props[k]['centroid_pan'] for k in centroid_keys])

    is_pocket = []
    for fi, face in enumerate(ps_faces):
        fc = ps_verts[face].mean(axis=0)
        dists = np.linalg.norm(centroid_arr - fc, axis=1)
        nearest = centroid_keys[np.argmin(dists)]
        # Both Pan AND Groves faces near outer-ring notes are pocket
        # (the whole notepad + grove must be depressed to receive the notepad)
        is_pocket.append(nearest in section_notes)

    pocket_count = sum(is_pocket)
    print(f"  Pocket: {pocket_count}, rim: {len(is_pocket) - pocket_count}")

    # Cut holes in playing surface only
    print(f"  Cutting holes for {len(section_notes)} outer-ring notes...")
    for nname in sorted(section_notes):
        if nname not in note_props:
            continue
        props = note_props[nname]
        centroid = props['centroid_pan']
        normal = props['normal']
        bosses = props['boss_positions_pan']

        ps_faces, is_pocket = remove_faces_in_circle(
            ps_verts, ps_faces, is_pocket,
            centroid, normal, MOUNT_CLEARANCE_HOLE)

        for bp in bosses:
            ps_faces, is_pocket = remove_faces_in_circle(
                ps_verts, ps_faces, is_pocket,
                bp, normal, BOSS_THROUGH_HOLE)

        print(f"    {nname}: mount + {len(bosses)} boss holes")

    print(f"  After holes: {len(ps_faces)} faces")
    ps_verts, ps_faces = reindex_mesh(ps_verts, ps_faces)

    # DON'T thicken playing surface here — thicken per-section after angular cut
    # to avoid boundary side-faces at R~225mm spanning across sector boundaries.
    print(f"  Playing surface ready: {len(ps_verts)} verts, {len(ps_faces)} faces (thickened per-section)")

    # Compute global Y cap from playing surface vertices INSIDE the drum wall
    # transition zone (R < DRUM_WALL_RADIUS) so we clip at the actual playing
    # surface level, not the drum rim
    ps_radii = np.sqrt(ps_verts[:, 0]**2 + ps_verts[:, 2]**2)
    interior_mask = ps_radii < DRUM_WALL_RADIUS
    if interior_mask.any():
        global_y_cap = float(ps_verts[interior_mask, 1].max())
    else:
        global_y_cap = float(ps_verts[:, 1].max())
    global_ps_max_r = float(ps_radii.max())
    print(f"  Global Y cap: {global_y_cap:.1f}mm (playing surface), max_r: {global_ps_max_r:.0f}mm")

    # --- Drum wall: just reindex, do NOT thicken yet ---
    dw_verts, dw_faces = reindex_mesh(bowl_verts, dw_faces)
    print(f"  Drum wall surface: {len(dw_verts)} verts, {len(dw_faces)} faces (thickened per-section)")

    return ps_verts, ps_faces, is_pocket, dw_verts, dw_faces, global_y_cap, global_ps_max_r


# ============================================================
# Cut section from pre-built shells + add walls
# ============================================================

def extract_section(section_id, ps_surf_v, ps_surf_f, ps_is_pocket,
                    dw_surf_v, dw_surf_f,
                    note_props, output_dir,
                    global_y_cap, global_ps_max_r):
    """
    Cut one 60° section from both surfaces, thicken per-section,
    then combine with support structures.
    Uses global_y_cap and global_ps_max_r so all sections get identical supports.
    Profile sampling uses playing surface vertices (not full bowl with drum wall).
    """
    if section_id not in SECTIONS:
        print(f"ERROR: Unknown section '{section_id}'")
        return None

    angle_start, angle_end, note_names = SECTIONS[section_id]
    effective_end = angle_end if angle_end <= 360 else angle_end - 360
    span = normalize_angle(effective_end - angle_start)
    if span <= 0:
        span += 360
    angle_mid = angle_start + span / 2

    print(f"\n{'='*60}")
    print(f"Generating Section: {section_id}")
    print(f"  Angle range: {angle_start}° to {effective_end}°")
    print(f"  Notes: {note_names}")
    print(f"{'='*60}")

    # Cut playing surface by angle, then thicken per-section
    print(f"  Cutting playing surface...")
    ps_cut_faces, ps_cut_pocket = cut_quarter(
        ps_surf_v, ps_surf_f, angle_start, angle_end, ps_is_pocket)
    ps_v, ps_f = reindex_mesh(ps_surf_v, ps_cut_faces)
    print(f"    Playing surface: {len(ps_f)} faces")
    ps_v, ps_f = thicken_with_pocket(ps_v, ps_f, ps_cut_pocket,
                                      SECTOR_THICKNESS, POCKET_DEPTH)
    print(f"    Playing surface thickened: {len(ps_v)} verts, {len(ps_f)} faces")

    # Cut drum wall surface by angle, then thicken per-section
    print(f"  Cutting drum wall surface...")
    dw_cut_faces, = cut_quarter(dw_surf_v, dw_surf_f, angle_start, angle_end)
    dw_v, dw_f = reindex_mesh(dw_surf_v, dw_cut_faces)
    print(f"    Drum wall surface: {len(dw_f)} faces")
    # Thicken this section's drum wall piece
    dw_is_pocket = [True] * len(dw_f)
    dw_v, dw_f = thicken_with_pocket(dw_v, dw_f, dw_is_pocket, SECTOR_THICKNESS, 0.0)
    print(f"    Drum wall thickened: {len(dw_v)} verts, {len(dw_f)} faces")

    # Combine playing surface + drum wall
    n_ps = len(ps_v)
    s_verts = np.vstack([ps_v, dw_v]) if len(dw_v) > 0 else ps_v
    s_faces = list(ps_f) + [[vi + n_ps for vi in f] for f in dw_f]

    if not s_faces:
        print("  ERROR: No faces in section range!")
        return None

    # Use global values so all sections get identical support geometry
    ps_max_r = global_ps_max_r
    y_cap = global_y_cap
    base_y = s_verts[:, 1].min()

    print(f"  Building support walls (R: {WALL_INNER_RADIUS} to {ps_max_r:.0f}, Y cap: {y_cap:.1f})...")

    for cut_a in [angle_start, effective_end]:
        profile = sample_surface_profile(
            ps_surf_v, cut_a, WALL_INNER_RADIUS, ps_max_r)
        # Clip profile to playing surface level
        profile = [(r, min(y, y_cap)) for r, y in profile]

        # Build wall with cable routing slot
        wv, wf = build_wall_with_hole(
            cut_a, angle_mid, profile, base_y, WALL_THICKNESS)

        if len(wv) > 0:
            n_base = len(s_verts)
            s_verts = np.vstack([s_verts, wv])
            s_faces += [[vi + n_base for vi in f] for f in wf]

            # Bolt holes
            bv, bf = add_bolt_features(
                cut_a, angle_mid, profile, base_y,
                WALL_BOLT_SPACING, WALL_BOLT_DIAMETER, WALL_THICKNESS)
            if len(bv) > 0:
                n_base = len(s_verts)
                s_verts = np.vstack([s_verts, bv])
                s_faces += [[vi + n_base for vi in f] for f in bf]

            # Finger joints
            fv, ff = build_finger_joints(
                cut_a, angle_mid, profile, base_y, WALL_THICKNESS)
            if len(fv) > 0:
                n_base = len(s_verts)
                s_verts = np.vstack([s_verts, fv])
                s_faces += [[vi + n_base for vi in f] for f in ff]

            print(f"    Wall at {cut_a}°: slot + bolts + finger joints")

    # Inner arc support with studs and wire hole
    print(f"  Building inner arc support (R={INNER_CUTOFF_RADIUS}mm)...")
    # Sample surface height at inner radius for arc top, capped at playing surface
    arc_profile = sample_surface_profile(
        ps_surf_v, angle_mid,
        INNER_CUTOFF_RADIUS - 1, INNER_CUTOFF_RADIUS + 1, n_samples=1)
    arc_y_top = min(arc_profile[0][1], y_cap) if arc_profile else min(base_y + 30, y_cap)

    av, af = build_inner_arc_support(
        angle_start, effective_end, INNER_CUTOFF_RADIUS,
        arc_y_top, base_y)
    if len(av) > 0:
        n_base = len(s_verts)
        s_verts = np.vstack([s_verts, av])
        s_faces += [[vi + n_base for vi in f] for f in af]
        print(f"    Inner arc: {len(av)} verts, studs + wire hole")

    # Final stats
    bbox_min = s_verts.min(axis=0)
    bbox_max = s_verts.max(axis=0)
    bbox_size = bbox_max - bbox_min

    print(f"\n  Final mesh: {len(s_verts)} vertices, {len(s_faces)} faces")
    print(f"  Bounding box: {bbox_size[0]:.1f} x {bbox_size[1]:.1f} x {bbox_size[2]:.1f} mm")
    fits = all(d <= 256 for d in bbox_size)
    print(f"  Fits P1S (256mm): {'YES' if fits else 'NO (see dimensions)'}")

    # Output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    obj_path = output_dir / f"section_{section_id}.obj"
    stl_path = output_dir / f"section_{section_id}.stl"

    write_quarter_obj(obj_path, s_verts, s_faces, f"Section_{section_id}")
    write_quarter_stl(stl_path, s_verts, s_faces, f"Section_{section_id}")

    return {
        'section': section_id,
        'angle_start': angle_start,
        'angle_end': float(effective_end),
        'notes': note_names,
        'bbox_size': bbox_size.tolist(),
        'bbox_min': bbox_min.tolist(),
        'bbox_max': bbox_max.tolist(),
        'fits_p1s': fits,
        'n_vertices': len(s_verts),
        'n_faces': len(s_faces),
        'obj_path': str(obj_path),
        'stl_path': str(stl_path),
    }, s_verts, s_faces


# ============================================================
# Main
# ============================================================

def main():
    OBJ_PATH = "data/Tenor Pan only.obj"
    PROPS_PATH = "data/notepads/notepad_properties.json"
    OUTPUT_DIR = "data/quarters"

    args = sys.argv[1:]
    gen_all = '--all' in args
    orient_only = '--orient-only' in args
    specific = [a for a in args if not a.startswith('--')]

    if not specific and not gen_all:
        specific = ['S0']

    print("=" * 60)
    print("Pan Section Generator (Outer Ring — Sixths)")
    print("=" * 60)

    # Phase A: Load bowl surface
    print("\nPhase A: Loading bowl surface...")
    bowl_v, bowl_f, face_mat, face_grp, pan_offset = extract_bowl_surface(OBJ_PATH)

    # Phase B: Orient pan
    print("\nPhase B: Orienting pan (drum wall vertical along +Z)...")
    R = compute_pan_rotation(bowl_v, bowl_f, face_mat)
    bowl_v = bowl_v @ R.T

    bbox = bowl_v.max(axis=0) - bowl_v.min(axis=0)
    print(f"  Oriented bbox: {bbox[0]:.1f} x {bbox[1]:.1f} x {bbox[2]:.1f} mm")
    print(f"  X: [{bowl_v[:,0].min():.1f}, {bowl_v[:,0].max():.1f}]")
    print(f"  Y: [{bowl_v[:,1].min():.1f}, {bowl_v[:,1].max():.1f}]")
    print(f"  Z: [{bowl_v[:,2].min():.1f}, {bowl_v[:,2].max():.1f}]")

    if orient_only:
        out_path = Path(OUTPUT_DIR) / "pan_oriented.obj"
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        write_quarter_obj(out_path, bowl_v, bowl_f, "OrientedPan")
        print(f"\nOriented pan exported to: {out_path}")
        return

    # Phase C: Load notepad properties and rotate
    print("\nPhase C: Loading notepad properties...")
    note_props = load_notepad_properties(PROPS_PATH, pan_offset)
    apply_rotation_to_props(R, note_props)
    print(f"  Loaded and rotated {len(note_props)} notes")

    rotation_matrix = R.tolist()

    # Phase D: Prepare surfaces (playing surface + drum wall separately)
    print("\nPhase D: Preparing surfaces...")
    ps_surf_v, ps_surf_f, ps_is_pocket, dw_surf_v, dw_surf_f, \
        global_y_cap, global_ps_max_r = prepare_shells(
            bowl_v, bowl_f, face_mat, note_props)

    # Phase E: Generate each section from actual bowl geometry
    # Support structures (walls, inner arc) use global parameters for consistency
    sections = list(SECTIONS.keys()) if gen_all else specific

    print(f"\nPhase E: Cutting {len(sections)} section(s)...")
    results = []
    for sid in sections:
        gen_result = extract_section(
            sid, ps_surf_v, ps_surf_f, ps_is_pocket,
            dw_surf_v, dw_surf_f,
            note_props, OUTPUT_DIR,
            global_y_cap, global_ps_max_r)
        if gen_result:
            result, _, _ = gen_result
            results.append(result)

    # Save properties
    if results:
        output = {
            'rotation_matrix': rotation_matrix,
            'sections': results,
        }
        props_path = Path(OUTPUT_DIR) / "section_properties.json"
        with open(props_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved properties to: {props_path}")

    print(f"\n{'='*60}")
    print(f"Generated {len(results)} section(s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
