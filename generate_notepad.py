#!/usr/bin/env python3
"""
Generate printable note pad geometry from the tenor pan OBJ file.

Extracts note pad (Pan) and groove (Groves) geometry, converts from cm to mm,
and thickens into a solid object suitable for 3D printing.
"""

import numpy as np
from pathlib import Path
import math
import trimesh

# Source data URL — downloaded on demand if not present locally
SOURCE_OBJ_URL = "https://github.com/profLewis/deepPan/raw/main/data/Tenor%20Pan%20only.obj"
SOURCE_OBJ_PATH = "data/Tenor Pan only.obj"


def ensure_source_obj(path=SOURCE_OBJ_PATH):
    """Download the source OBJ if it doesn't exist locally."""
    if Path(path).exists():
        return path
    print(f"Source OBJ not found locally. Downloading from GitHub...")
    import urllib.request
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(SOURCE_OBJ_URL, path)
    print(f"  Downloaded: {path} ({Path(path).stat().st_size / 1e6:.1f} MB)")
    return path


# Conversion factor: OBJ file is in cm, output in mm
CM_TO_MM = 10.0

# Scale factor for the pan geometry (2x to make it bigger)
PAN_SCALE = 2.0

# Thickness for the solids (mm)
PAN_THICKNESS = 5.0       # Thickness of the pan playing surface (downward)
GROVE_DEPTH = 5.0         # Groove thickness downward (same as pan)
GROVE_PROTRUSION = 0.2    # Groove protrusion upward (small lip)
PAD_SHRINK = 0.3          # mm inward per edge — ensures pad fits in groove pocket

# Mounting cylinder parameters — OUTER ring (large sensor)
MOUNT_INNER_DIAMETER = 23.5   # Internal diameter
MOUNT_DEPTH = 14.0            # Cylinder depth (extended for better mount engagement)
MOUNT_WALL_THICKNESS = 2.5    # Wall thickness
MOUNT_THREAD_PITCH = 2.0      # Thread pitch
MOUNT_THREAD_DEPTH = 1.0      # Thread depth (outward from wall)
MOUNT_NOTCH_WIDTH = 2.0       # Wire notch width
MOUNT_SEGMENTS = 48           # Resolution for cylinder

# Anti-rotation rib parameters (on exterior of mount cylinder)
RIB_WIDTH = 2.0               # Width of rib in mm
RIB_HEIGHT = 1.0              # Protrusion outward in mm
RIB_ANGLE = math.pi           # Position: 180 degrees from notch

# Minimum pad size to accommodate cylinder (with margin)
MOUNT_OUTER_DIAMETER = MOUNT_INNER_DIAMETER + 2 * MOUNT_WALL_THICKNESS + 2 * MOUNT_THREAD_DEPTH
MIN_PAD_SIZE = MOUNT_OUTER_DIAMETER + 0.5  # Add 0.5mm margin

# Mounting cylinder parameters — CENTRAL ring (Adafruit 1740: 14mm dia, 2.5mm thick)
# Boss is sunk INTO the thickened pad body, sensor close to playing surface.
CENTRAL_MOUNT_INNER_DIAMETER = 16.0   # 14mm sensor + 2mm clearance (plenty of room)
CENTRAL_MOUNT_DEPTH = 12.0            # Boss bore depth

# The pad is thickened to contain the boss. Sensor sits near the top.
SENSOR_SKIN = 1.5                     # Material between sensor and playing surface
CENTRAL_PAD_THICKNESS = CENTRAL_MOUNT_DEPTH + SENSOR_SKIN + 1.0  # 12.5mm total
# (boss top at 1.5mm below surface, extends 10mm down, 1mm pad floor below boss)

# Sensor pocket is no longer needed as a separate cut — the boss bore
# itself brings the sensor to SENSOR_SKIN mm from the surface.
SENSOR_POCKET_DIAMETER = 14.5         # kept for reference
SENSOR_POCKET_DEPTH = 0.0             # disabled — boss position handles this
CENTRAL_MOUNT_WALL_THICKNESS = 2.0    # Wall thickness
CENTRAL_MOUNT_THREAD_PITCH = 2.0      # Push-fit thread pitch (internal ridges)
CENTRAL_MOUNT_THREAD_DEPTH = 0.3      # Shallow push-fit ridges (easy in/out)
CENTRAL_MOUNT_NOTCH_WIDTH = 2.0       # Wire notch width
CENTRAL_MOUNT_OUTER_DIAMETER = CENTRAL_MOUNT_INNER_DIAMETER + 2 * CENTRAL_MOUNT_WALL_THICKNESS
CENTRAL_MIN_PAD_SIZE = CENTRAL_MOUNT_OUTER_DIAMETER + 0.5

# Mounting cylinder parameters — INNER ring (same sensor, same deep boss)
SMALL_MOUNT_INNER_DIAMETER = CENTRAL_MOUNT_INNER_DIAMETER
SMALL_MOUNT_DEPTH = CENTRAL_MOUNT_DEPTH
SMALL_MOUNT_WALL_THICKNESS = CENTRAL_MOUNT_WALL_THICKNESS
SMALL_MOUNT_THREAD_PITCH = CENTRAL_MOUNT_THREAD_PITCH
SMALL_MOUNT_THREAD_DEPTH = CENTRAL_MOUNT_THREAD_DEPTH
SMALL_MOUNT_NOTCH_WIDTH = CENTRAL_MOUNT_NOTCH_WIDTH
SMALL_MOUNT_OUTER_DIAMETER = CENTRAL_MOUNT_OUTER_DIAMETER
SMALL_MIN_PAD_SIZE = CENTRAL_MIN_PAD_SIZE

# M2 through-hole parameters (holes through full pad thickness)
SCREW_HOLE_DIAMETER = 2.2     # M2 clearance hole (2.2mm for M2 bolt)
SCREW_HOLE_INSET = 8.0        # Distance inset from pad boundary
SCREW_HOLE_SEGMENTS = 16      # Resolution for hole cylinders
SCREW_HOLE_COUNT = 4          # Number of holes per pad

# M2 tapered countersink (recess for M2x8 flat-head screw)
COUNTERSINK_TOP_DIA = 4.0     # M2 flat head ~3.8mm + 0.2mm clearance
COUNTERSINK_BOT_DIA = SCREW_HOLE_DIAMETER  # tapers to shaft clearance (2.2mm)
COUNTERSINK_DEPTH = 1.0       # ~1mm taper depth (90° DIN 965)
PLUG_BORE_DEPTH = 2.0         # mm — cylindrical bore above taper for flush plug

# Cap dimensions (press-fit plug to cover countersink + plug bore)
CAP_CLEARANCE = 0.1           # mm clearance for press fit
CAP_NEEDLE_HOLE = 0.8         # mm needle hole for cap removal

# Legacy aliases for any code still referencing counterbore
COUNTERBORE_DIAMETER = COUNTERSINK_TOP_DIA
COUNTERBORE_DEPTH = COUNTERSINK_DEPTH

# Aliases used by generate_sector.py / generate_quarter.py for standoff sizing
BOSS_HOLE_DIAMETER = SCREW_HOLE_DIAMETER      # M2 clearance
BOSS_HEIGHT = PAN_THICKNESS                    # Through-hole: full pad thickness
BOSS_OUTER_DIAMETER = 6.0                      # Standoff OD (sized for M2)

# Note mapping: (grove_object, pan_object) -> (index, note, ring, octave)
NOTE_MAPPING = {
    # Outer Ring (4ths)
    ('object_58', 'object_62'): ('O0', 'F#', 'outer', 4),
    ('object_57', 'object_63'): ('O1', 'B', 'outer', 4),
    ('object_56', 'object_64'): ('O2', 'E', 'outer', 4),
    ('object_55', 'object_90'): ('O3', 'A', 'outer', 4),
    ('object_54', 'object_65'): ('O4', 'D', 'outer', 4),
    ('object_53', 'object_66'): ('O5', 'G', 'outer', 4),
    ('object_59', 'object_60'): ('O6', 'C', 'outer', 4),
    ('object_52', 'object_61'): ('O7', 'F', 'outer', 4),
    ('object_51', 'object_67'): ('O8', 'Bb', 'outer', 4),
    ('object_50', 'object_88'): ('O9', 'Eb', 'outer', 4),
    ('object_49', 'object_68'): ('O10', 'Ab', 'outer', 4),
    ('object_48', 'object_69'): ('O11', 'C#', 'outer', 4),
    # Central Ring (5ths)
    ('object_25', 'object_45'): ('C0', 'F#', 'central', 5),
    ('object_24', 'object_46'): ('C1', 'B', 'central', 5),
    ('object_23', 'object_47'): ('C2', 'E', 'central', 5),
    ('object_22', 'object_37'): ('C3', 'A', 'central', 5),
    ('object_21', 'object_38'): ('C4', 'D', 'central', 5),
    ('object_20', 'object_39'): ('C5', 'G', 'central', 5),
    ('object_73', 'object_31'): ('C6', 'C', 'central', 5),
    ('object_30', 'object_40'): ('C7', 'F', 'central', 5),
    ('object_29', 'object_41'): ('C8', 'Bb', 'central', 5),
    ('object_28', 'object_42'): ('C9', 'Eb', 'central', 5),
    ('object_27', 'object_43'): ('C10', 'Ab', 'central', 5),
    ('object_26', 'object_44'): ('C11', 'C#', 'central', 5),
    # Inner Ring (6ths)
    ('object_72', 'object_36'): ('I0', 'C#', 'inner', 6),
    ('object_71', 'object_32'): ('I1', 'E', 'inner', 6),
    ('object_19', 'object_33'): ('I2', 'D', 'inner', 6),
    ('object_70', 'object_34'): ('I3', 'C', 'inner', 6),
    ('object_18', 'object_35'): ('I4', 'Eb', 'inner', 6),
}

# Reverse lookup: index -> (grove_object, pan_object, note, ring, octave)
NOTE_BY_INDEX = {}
for (grove, pan), (idx, note, ring, octave) in NOTE_MAPPING.items():
    NOTE_BY_INDEX[idx] = {
        'grove_object': grove,
        'pan_object': pan,
        'note': note,
        'ring': ring,
        'octave': octave
    }


def compute_leveling_rotation(obj_path):
    """
    Compute the rotation matrix that levels the pan (same as generate_pan_body.py).

    Fits a plane to the Groves vertices, finds the drum axis, and returns
    the rotation matrix R that aligns it with +Y.
    """
    objects, all_vertices = parse_obj_file(obj_path)

    # Collect Groves vertices
    groves_vi = set()
    for obj_name, obj_data in objects.items():
        if obj_data.get('material') == 'Groves':
            groves_vi.update(obj_data['face_vertices'])

    groves_verts = all_vertices[sorted(groves_vi)] * CM_TO_MM * PAN_SCALE

    centered = groves_verts - groves_verts.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    drum_axis = Vt[2]
    if drum_axis[1] < 0:
        drum_axis = -drum_axis

    target = np.array([0.0, 1.0, 0.0])
    if np.allclose(drum_axis, target, atol=1e-6):
        return np.eye(3)

    v = np.cross(drum_axis, target)
    s = np.linalg.norm(v)
    c = np.dot(drum_axis, target)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    return R


def parse_obj_file(filepath):
    """Parse OBJ file and extract geometry."""
    objects = {}
    current_object = None
    current_material = None
    all_vertices = []

    with open(filepath, 'r') as f:
        content = f.read()
        content = content.replace('\\\r\n', ' ').replace('\\\n', ' ')
        lines = content.split('\n')

    for line in lines:
        line = line.strip()
        if line.startswith('g '):
            current_object = line[2:]
        elif line.startswith('usemtl '):
            current_material = line[7:]
            if current_object:
                objects[current_object] = {
                    'material': current_material,
                    'face_vertices': set(),
                    'faces': []
                }
        elif line.startswith('v '):
            parts = line.split()
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            all_vertices.append((x, y, z))
        elif line.startswith('f ') and current_object in objects:
            parts = line.split()[1:]
            face_verts = []
            for p in parts:
                try:
                    v_idx = int(p.split('/')[0]) - 1
                    objects[current_object]['face_vertices'].add(v_idx)
                    face_verts.append(v_idx)
                except ValueError:
                    continue
            if face_verts:
                objects[current_object]['faces'].append(face_verts)

    return objects, np.array(all_vertices)


def extract_object_mesh(objects, obj_name, all_vertices):
    """Extract vertices and faces for an object, re-indexed."""
    obj = objects[obj_name]
    old_indices = sorted(list(obj['face_vertices']))

    # Create mapping from old to new indices
    index_map = {old: new for new, old in enumerate(old_indices)}

    # Extract vertices (convert cm to mm and apply scale factor)
    vertices = all_vertices[old_indices] * CM_TO_MM * PAN_SCALE

    # Re-index faces
    faces = []
    for face in obj['faces']:
        new_face = [index_map[idx] for idx in face if idx in index_map]
        if len(new_face) >= 3:
            faces.append(new_face)

    return vertices, faces


def compute_vertex_normals(vertices, faces):
    """Compute per-vertex normals by averaging face normals."""
    vertex_normals = np.zeros_like(vertices)

    for face in faces:
        if len(face) < 3:
            continue
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(face_normal)
        if norm > 0:
            face_normal = face_normal / norm

        # Add to all vertices in the face
        for idx in face:
            vertex_normals[idx] += face_normal

    # Normalize
    for i in range(len(vertex_normals)):
        norm = np.linalg.norm(vertex_normals[i])
        if norm > 0:
            vertex_normals[i] = vertex_normals[i] / norm
        else:
            vertex_normals[i] = np.array([0, -1, 0])  # Default downward

    return vertex_normals


def compute_surface_normal(vertices, faces):
    """Compute the average surface normal weighted by face area."""
    total_normal = np.zeros(3)
    total_area = 0

    for face in faces:
        if len(face) < 3:
            continue
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = np.cross(edge1, edge2)
        area = np.linalg.norm(cross) / 2
        if area > 0:
            normal = cross / (2 * area)
            total_normal += normal * area
            total_area += area

    if total_area > 0:
        total_normal = total_normal / np.linalg.norm(total_normal)
    else:
        total_normal = np.array([0, -1, 0])

    return total_normal


def compute_interior_centroid(vertices, faces, normal, thickness_down, thickness_up=0):
    """
    Compute a centroid guaranteed to be inside the thickened volume.

    Takes the surface centroid and offsets it to the middle of the thickness.
    """
    # Surface centroid (average of all vertices)
    surface_centroid = vertices.mean(axis=0)

    # The volume extends from (surface - thickness_down*normal) to (surface + thickness_up*normal)
    # The middle of this range is: surface + (thickness_up - thickness_down)/2 * normal
    offset = (thickness_up - thickness_down) / 2.0
    interior_centroid = surface_centroid + normal * offset

    return interior_centroid, surface_centroid


def generate_threaded_mount_cylinder(inner_diameter, depth, wall_thickness, thread_pitch,
                                      thread_depth, notch_width, segments=MOUNT_SEGMENTS,
                                      threads_inside=False):
    """
    Generate a solid threaded mounting cylinder with a wire notch properly cut out.

    The cylinder is centered at origin, extending downward along -Z axis.
    The notch is cut through the wall at angle=0 (positive X direction).

    threads_inside=False: threads on outer surface (outer ring, for mount base)
    threads_inside=True:  threads on inner bore (central/inner ring, for push-cap)

    Returns vertices and faces for a watertight mesh.
    """
    inner_r = inner_diameter / 2
    outer_r = inner_r + wall_thickness

    vertices = []
    faces = []

    notch_half_width = notch_width / 2

    # Anti-rotation rib only for external threads (outer ring)
    rib_half_angle = math.atan2(RIB_WIDTH / 2, outer_r)

    def in_rib(seg_idx):
        if threads_inside:
            return False  # no rib for internal-thread bosses
        angle = 2 * math.pi * seg_idx / segments
        angle_diff = abs(angle - RIB_ANGLE)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        return angle_diff <= rib_half_angle

    # Calculate notch angle (use the angle that gives notch_half_width at inner radius)
    notch_half_angle = math.atan2(notch_half_width, inner_r)

    # Number of segments to skip for notch (at least 2)
    notch_segments = max(2, int(notch_half_angle * 2 * segments / (2 * math.pi)) + 1)

    # Make notch symmetric around segment 0
    # Skip segments from (segments - notch_segments//2) to (notch_segments//2)
    notch_start = segments - notch_segments // 2
    notch_end = notch_segments - notch_segments // 2

    def in_notch(seg_idx):
        return seg_idx >= notch_start or seg_idx < notch_end

    # Number of Z levels (16 per pitch for smooth helical thread)
    z_levels = max(int(depth / thread_pitch * 16), 32)

    # Generate cylinder vertices (excluding notch segments)
    inner_rings = []
    outer_rings = []

    for z_idx in range(z_levels + 1):
        z = -z_idx * depth / z_levels

        inner_ring = {}
        outer_ring = {}

        # Push-fit thread profile: gentle sawtooth for easy push in/out
        thread_phase = (z_idx / z_levels * depth / thread_pitch) % 1.0
        if threads_inside:
            # Asymmetric sawtooth: 70% gentle ramp, 30% steep retention
            if thread_phase < 0.7:
                thread_h = thread_depth * (thread_phase / 0.7)
            else:
                thread_h = thread_depth * (1.0 - (thread_phase - 0.7) / 0.3)
        else:
            # Symmetric triangle for external threads
            thread_h = thread_depth * (1 - abs(2 * thread_phase - 1))

        for i in range(segments):
            if in_notch(i):
                continue

            angle = 2 * math.pi * i / segments

            if threads_inside:
                # Internal threads: ridges protrude inward from bore
                ir = inner_r - thread_h
                inner_ring[i] = len(vertices)
                vertices.append([ir * math.cos(angle), ir * math.sin(angle), z])
                # Outer wall: smooth (no external threads)
                outer_ring[i] = len(vertices)
                vertices.append([outer_r * math.cos(angle), outer_r * math.sin(angle), z])
            else:
                # External threads: ridges protrude outward from wall
                inner_ring[i] = len(vertices)
                vertices.append([inner_r * math.cos(angle), inner_r * math.sin(angle), z])
                r = outer_r + thread_h + (RIB_HEIGHT if in_rib(i) else 0)
                outer_ring[i] = len(vertices)
                vertices.append([r * math.cos(angle), r * math.sin(angle), z])

        inner_rings.append(inner_ring)
        outer_rings.append(outer_ring)

    # Notch boundary angles (first and last non-notch segments)
    valid_segs = sorted([i for i in range(segments) if not in_notch(i)])
    first_valid = valid_segs[0]
    last_valid = valid_segs[-1]

    first_angle = 2 * math.pi * first_valid / segments
    last_angle = 2 * math.pi * last_valid / segments

    # Notch wall vertices (at the edges of the notch)
    # Left wall at last_valid angle, right wall at first_valid angle
    notch_inner_left = []  # at last_angle
    notch_outer_left = []
    notch_inner_right = []  # at first_angle
    notch_outer_right = []

    # Notch wall outer radius: smooth wall (no threads at the notch edge)
    notch_outer_r = outer_r if threads_inside else outer_r + thread_depth

    for z_idx in range(z_levels + 1):
        z = -z_idx * depth / z_levels

        # Left edge (at last_angle, which is just before the notch going clockwise)
        notch_inner_left.append(len(vertices))
        vertices.append([inner_r * math.cos(last_angle), inner_r * math.sin(last_angle), z])
        notch_outer_left.append(len(vertices))
        vertices.append([notch_outer_r * math.cos(last_angle), notch_outer_r * math.sin(last_angle), z])

        # Right edge (at first_angle, which is just after the notch going clockwise)
        notch_inner_right.append(len(vertices))
        vertices.append([inner_r * math.cos(first_angle), inner_r * math.sin(first_angle), z])
        notch_outer_right.append(len(vertices))
        vertices.append([notch_outer_r * math.cos(first_angle), notch_outer_r * math.sin(first_angle), z])

    # === BUILD FACES ===

    # Top cap - connect inner to outer for each segment pair
    for idx in range(len(valid_segs) - 1):
        i = valid_segs[idx]
        i_next = valid_segs[idx + 1]
        faces.append([inner_rings[0][i], outer_rings[0][i],
                     outer_rings[0][i_next], inner_rings[0][i_next]])

    # Top cap notch closure (connects main cap to notch edges)
    faces.append([notch_inner_left[0], notch_outer_left[0],
                 outer_rings[0][last_valid], inner_rings[0][last_valid]])
    faces.append([inner_rings[0][first_valid], outer_rings[0][first_valid],
                 notch_outer_right[0], notch_inner_right[0]])
    # NOTE: No face across notch opening - it's an open slot for wires

    # Bottom cap
    for idx in range(len(valid_segs) - 1):
        i = valid_segs[idx]
        i_next = valid_segs[idx + 1]
        faces.append([inner_rings[-1][i], inner_rings[-1][i_next],
                     outer_rings[-1][i_next], outer_rings[-1][i]])

    # Bottom cap notch closure (connects main cap to notch edges)
    faces.append([notch_inner_left[-1], inner_rings[-1][last_valid],
                 outer_rings[-1][last_valid], notch_outer_left[-1]])
    faces.append([inner_rings[-1][first_valid], notch_inner_right[-1],
                 notch_outer_right[-1], outer_rings[-1][first_valid]])
    # NOTE: No face across notch opening - it's an open slot for wires

    # Inner wall (cylinder bore)
    for z_idx in range(z_levels):
        for idx in range(len(valid_segs) - 1):
            i = valid_segs[idx]
            i_next = valid_segs[idx + 1]
            faces.append([inner_rings[z_idx][i], inner_rings[z_idx][i_next],
                         inner_rings[z_idx+1][i_next], inner_rings[z_idx+1][i]])

    # Outer wall (threaded surface)
    for z_idx in range(z_levels):
        for idx in range(len(valid_segs) - 1):
            i = valid_segs[idx]
            i_next = valid_segs[idx + 1]
            faces.append([outer_rings[z_idx][i], outer_rings[z_idx+1][i],
                         outer_rings[z_idx+1][i_next], outer_rings[z_idx][i_next]])

    # Notch left wall (radial face at last_angle)
    for z_idx in range(z_levels):
        faces.append([notch_inner_left[z_idx], inner_rings[z_idx][last_valid],
                     inner_rings[z_idx+1][last_valid], notch_inner_left[z_idx+1]])
        faces.append([outer_rings[z_idx][last_valid], notch_outer_left[z_idx],
                     notch_outer_left[z_idx+1], outer_rings[z_idx+1][last_valid]])
        faces.append([notch_inner_left[z_idx], notch_outer_left[z_idx],
                     notch_outer_left[z_idx+1], notch_inner_left[z_idx+1]])

    # Notch right wall (radial face at first_angle)
    for z_idx in range(z_levels):
        faces.append([inner_rings[z_idx][first_valid], notch_inner_right[z_idx],
                     notch_inner_right[z_idx+1], inner_rings[z_idx+1][first_valid]])
        faces.append([notch_outer_right[z_idx], outer_rings[z_idx][first_valid],
                     outer_rings[z_idx+1][first_valid], notch_outer_right[z_idx+1]])
        faces.append([notch_outer_right[z_idx], notch_inner_right[z_idx],
                     notch_inner_right[z_idx+1], notch_outer_right[z_idx+1]])

    return np.array(vertices), faces


def check_and_scale_pad(pan_verts, grove_verts, min_size=MIN_PAD_SIZE):
    """
    Check if the pad is large enough for the mounting cylinder.
    If not, scale it up uniformly in the XZ plane (preserving Y/thickness).

    The cylinder will be placed at the centroid, so we need the pad to be
    at least min_size in both X and Z dimensions around that center.

    Returns:
        scaled_pan_verts, scaled_grove_verts, scale_factor, was_scaled
    """
    # Use pan vertices to determine size (pan is the primary surface)
    # The cylinder is placed at the pan centroid

    x_extent = pan_verts[:, 0].max() - pan_verts[:, 0].min()
    z_extent = pan_verts[:, 2].max() - pan_verts[:, 2].min()
    min_extent = min(x_extent, z_extent)

    if min_extent >= min_size:
        # No scaling needed
        return pan_verts, grove_verts, 1.0, False

    # Calculate scale factor needed
    scale_factor = min_size / min_extent

    # Find centroid for scaling (scale from pan center)
    center_x = (pan_verts[:, 0].max() + pan_verts[:, 0].min()) / 2
    center_z = (pan_verts[:, 2].max() + pan_verts[:, 2].min()) / 2

    # Scale pan vertices (XZ only, preserve Y)
    scaled_pan = pan_verts.copy()
    scaled_pan[:, 0] = center_x + (pan_verts[:, 0] - center_x) * scale_factor
    scaled_pan[:, 2] = center_z + (pan_verts[:, 2] - center_z) * scale_factor

    # Scale grove vertices from same center
    scaled_grove = grove_verts.copy()
    scaled_grove[:, 0] = center_x + (grove_verts[:, 0] - center_x) * scale_factor
    scaled_grove[:, 2] = center_z + (grove_verts[:, 2] - center_z) * scale_factor

    return scaled_pan, scaled_grove, scale_factor, True


def transform_cylinder_to_normal(cylinder_verts, centroid, normal):
    """
    Transform cylinder from Z-down orientation to align with given normal.
    Cylinder is moved so its top center is at the centroid.
    """
    # Default cylinder points down along -Z
    # We need to rotate so -Z aligns with -normal (cylinder extends away from surface)

    z_axis = np.array([0, 0, -1])
    target = -normal / np.linalg.norm(normal)

    # Rotation matrix from z_axis to target
    v = np.cross(z_axis, target)
    c = np.dot(z_axis, target)

    if np.linalg.norm(v) < 1e-10:
        if c > 0:
            rot_matrix = np.eye(3)
        else:
            # 180 degree rotation around X axis
            rot_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    else:
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rot_matrix = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))

    # Apply rotation then translation
    rotated = cylinder_verts @ rot_matrix.T
    translated = rotated + centroid

    return translated


def compute_cylinder_surface_offset(pan_verts, centroid, normal, cylinder_radius):
    """
    Compute how much to lower the cylinder to avoid protrusions through the pan surface.

    The pan surface is curved, so the surface height at the cylinder footprint edge
    may be lower than at the centroid. We find the minimum surface height within
    the cylinder's footprint and return the offset needed to lower the cylinder.

    Returns: offset amount (negative means lower the cylinder)
    """
    # Create local coordinate system with normal as Z
    z_local = normal / np.linalg.norm(normal)

    # Find perpendicular axes
    if abs(z_local[0]) < 0.9:
        x_local = np.cross(z_local, np.array([1, 0, 0]))
    else:
        x_local = np.cross(z_local, np.array([0, 1, 0]))
    x_local = x_local / np.linalg.norm(x_local)
    y_local = np.cross(z_local, x_local)

    # Transform pan vertices to local coordinates (relative to centroid)
    # In local coords: centroid is at origin, normal points along +Z
    rel_verts = pan_verts - centroid
    local_x = rel_verts @ x_local
    local_y = rel_verts @ y_local
    local_z = rel_verts @ z_local

    # Find vertices within cylinder radius (with small margin)
    radial_dist = np.sqrt(local_x**2 + local_y**2)
    within_radius = radial_dist <= cylinder_radius * 1.1  # 10% margin

    if not np.any(within_radius):
        return 0.0  # No adjustment needed

    # Find minimum Z (lowest surface point within cylinder footprint)
    min_z = local_z[within_radius].min()

    # If min_z is negative, the surface dips below the centroid plane
    # We should lower the cylinder top to this level (plus small margin)
    if min_z < 0:
        return min_z - 0.5  # Extra 0.5mm margin

    return 0.0


def find_boundary_edges(faces):
    """Find boundary edges (edges that belong to only one face)."""
    edge_count = {}
    edge_face_order = {}  # Track the order vertices appear in faces

    for face in faces:
        n = len(face)
        for i in range(n):
            v1, v2 = face[i], face[(i + 1) % n]
            edge = tuple(sorted([v1, v2]))
            edge_count[edge] = edge_count.get(edge, 0) + 1
            # Store the directed edge for winding order
            if edge not in edge_face_order:
                edge_face_order[edge] = (v1, v2)

    # Boundary edges appear exactly once
    boundary_edges = [(edge, edge_face_order[edge]) for edge, count in edge_count.items() if count == 1]
    return boundary_edges


def find_all_boundary_loops(edges_with_order):
    """Find all boundary loops using directed edge traversal.

    Uses the face-winding-consistent directed edges from find_boundary_edges()
    to guarantee correct loop ordering and consistent winding for side walls.
    """
    if not edges_with_order:
        return []

    # Build directed next-vertex map from face winding.
    # For boundary edges (appearing in exactly 1 face), the directed edge
    # (v1->v2) as stored by find_boundary_edges gives the correct winding.
    directed_next = {}
    for edge, (v1, v2) in edges_with_order:
        directed_next[v1] = v2

    all_loops = []
    visited = set()

    for start in directed_next:
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        current = directed_next.get(start)
        while current is not None and current != start and current not in visited:
            loop.append(current)
            visited.add(current)
            current = directed_next.get(current)
        if len(loop) >= 3 and current == start:
            all_loops.append(loop)
        elif len(loop) >= 3:
            # Open chain (non-manifold topology) — still useful, mark visited
            all_loops.append(loop)

    return all_loops


def validate_manifold(vertices, faces, label="mesh"):
    """Check mesh for non-manifold edges and report issues.

    Returns (is_ok, report_dict) where is_ok is True if the mesh has no
    non-manifold edges (edges shared by 3+ faces).
    """
    edge_count = {}
    for face in faces:
        n = len(face)
        for i in range(n):
            v1, v2 = face[i], face[(i + 1) % n]
            edge = tuple(sorted([v1, v2]))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    non_manifold = {e: c for e, c in edge_count.items() if c > 2}
    boundary = {e: c for e, c in edge_count.items() if c == 1}
    interior = {e: c for e, c in edge_count.items() if c == 2}

    is_ok = len(non_manifold) == 0
    report = {
        'total_edges': len(edge_count),
        'interior_edges': len(interior),
        'boundary_edges': len(boundary),
        'non_manifold_edges': len(non_manifold),
    }

    if not is_ok:
        print(f"  WARNING [{label}]: {len(non_manifold)} non-manifold edges "
              f"(shared by 3+ faces)")
    return is_ok, report


def thicken_surface(vertices, faces, thickness_down, thickness_up=0):
    """
    Thicken a surface mesh into a solid by:
    1. Creating offset vertices along normals (top and bottom surfaces)
    2. Creating side walls between all boundary edges

    Args:
        vertices: Original surface vertices
        faces: Original surface faces
        thickness_down: How much to extrude downward (along -normal)
        thickness_up: How much to extrude upward (along +normal), default 0
    """
    # Compute vertex normals
    normals = compute_vertex_normals(vertices, faces)

    # Create top vertices (offset upward if thickness_up > 0, else same as original)
    if thickness_up > 0:
        top_vertices = vertices + normals * thickness_up
    else:
        top_vertices = vertices.copy()

    # Create bottom vertices (offset downward)
    bottom_vertices = vertices - normals * thickness_down

    # Combine vertices: [top_vertices, bottom_vertices]
    n_verts = len(vertices)
    all_vertices = np.vstack([top_vertices, bottom_vertices])

    # Top faces (original, keep winding)
    top_faces = [list(face) for face in faces]

    # Bottom faces (offset indices, reverse winding for correct normals)
    bottom_faces = [[idx + n_verts for idx in reversed(face)] for face in faces]

    # Find all boundary loops and create side walls.
    # The directed traversal in find_all_boundary_loops guarantees that the
    # loop winds consistently with the top-face winding.  For outward-facing
    # side walls we need quads whose normal points away from the mesh interior.
    # With CCW top faces, the boundary loop traverses v1->v2 such that the
    # interior is on the LEFT.  The correct side quad is then:
    #   [v2, v1, v1+n_verts, v2+n_verts]  (outward-facing)
    boundary_edges = find_boundary_edges(faces)
    boundary_loops = find_all_boundary_loops(boundary_edges)

    side_faces = []
    for loop in boundary_loops:
        for i in range(len(loop)):
            v1 = loop[i]
            v2 = loop[(i + 1) % len(loop)]
            # Side quad: reversed order so normal faces outward
            side_faces.append([v2, v1, v1 + n_verts, v2 + n_verts])

    all_faces = top_faces + bottom_faces + side_faces

    # Validate manifold quality
    validate_manifold(all_vertices, all_faces, label="thicken_surface")

    return all_vertices, all_faces


def center_mesh(vertices):
    """Center mesh at origin and return the offset used."""
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    return centered, centroid


def write_obj(filepath, vertices, faces, object_name="NotePad"):
    """Write mesh to OBJ file."""
    with open(filepath, 'w') as f:
        f.write(f"# Note Pad Geometry for 3D Printing\n")
        f.write(f"# Generated by generate_notepad.py\n")
        f.write(f"# Units: mm\n\n")

        f.write(f"o {object_name}\n")

        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        f.write("\n")

        # Write faces (1-indexed)
        for face in faces:
            face_str = " ".join(str(idx + 1) for idx in face)
            f.write(f"f {face_str}\n")

    print(f"Saved: {filepath}")


def write_stl(filepath, vertices, faces, object_name="NotePad"):
    """Write mesh to binary STL file for 3D printing."""
    import struct

    def compute_face_normal(v0, v1, v2):
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        return normal

    # Triangulate faces (split quads into triangles)
    triangles = []
    for face in faces:
        if len(face) == 3:
            triangles.append(face)
        elif len(face) == 4:
            # Split quad into two triangles
            triangles.append([face[0], face[1], face[2]])
            triangles.append([face[0], face[2], face[3]])
        elif len(face) > 4:
            # Fan triangulation
            for i in range(1, len(face) - 1):
                triangles.append([face[0], face[i], face[i + 1]])

    with open(filepath, 'wb') as f:
        # Header (80 bytes)
        header = f"STL generated by generate_notepad.py - {object_name}".encode('ascii')
        header = header[:80].ljust(80, b'\0')
        f.write(header)

        # Number of triangles
        f.write(struct.pack('<I', len(triangles)))

        # Write triangles
        for tri in triangles:
            v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            normal = compute_face_normal(v0, v1, v2)

            # Normal
            f.write(struct.pack('<3f', normal[0], normal[1], normal[2]))
            # Vertices
            f.write(struct.pack('<3f', v0[0], v0[1], v0[2]))
            f.write(struct.pack('<3f', v1[0], v1[1], v1[2]))
            f.write(struct.pack('<3f', v2[0], v2[1], v2[2]))
            # Attribute byte count
            f.write(struct.pack('<H', 0))

    print(f"Saved: {filepath}")


def _create_frustum(r_top, r_bot, height, segments=24):
    """Create a truncated cone (frustum) as a trimesh."""
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    top_v = np.column_stack([r_top * cos_a, r_top * sin_a, np.full(segments, height / 2)])
    bot_v = np.column_stack([r_bot * cos_a, r_bot * sin_a, np.full(segments, -height / 2)])
    ct = np.array([[0, 0, height / 2]])
    cb = np.array([[0, 0, -height / 2]])
    verts = np.vstack([top_v, bot_v, ct, cb])
    faces = []
    ic_t, ic_b = 2 * segments, 2 * segments + 1
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([i, j, segments + j])
        faces.append([i, segments + j, segments + i])
        faces.append([ic_t, i, j])
        faces.append([ic_b, segments + j, segments + i])
    return trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)


def _create_screw_tool(shaft_r, head_r, total_h, taper_h, plug_h=PLUG_BORE_DEPTH, segments=24):
    """
    Create compound screw hole tool: shaft + taper + plug bore.

    Profile (bottom to top):
      - Cylindrical shaft (shaft_r) from z_bot to z_taper
      - Tapered countersink (shaft_r → head_r) from z_taper to z_plug
      - Cylindrical plug bore (head_r) from z_plug to z_top

    Single watertight mesh for one boolean operation.
    """
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    z_bot = -total_h / 2
    z_taper = total_h / 2 - plug_h - taper_h   # taper starts here
    z_plug = total_h / 2 - plug_h               # plug bore starts here
    z_top = total_h / 2
    # Ring 0: shaft bottom
    v0 = np.column_stack([shaft_r * cos_a, shaft_r * sin_a, np.full(segments, z_bot)])
    # Ring 1: shaft top / taper bottom (shaft radius)
    v1 = np.column_stack([shaft_r * cos_a, shaft_r * sin_a, np.full(segments, z_taper)])
    # Ring 2: taper top / plug bore bottom (head radius)
    v2 = np.column_stack([head_r * cos_a, head_r * sin_a, np.full(segments, z_plug)])
    # Ring 3: plug bore top (head radius)
    v3 = np.column_stack([head_r * cos_a, head_r * sin_a, np.full(segments, z_top)])
    ct = np.array([[0, 0, z_top]])
    cb = np.array([[0, 0, z_bot]])
    verts = np.vstack([v0, v1, v2, v3, ct, cb])
    s = segments
    ic_t, ic_b = 4 * s, 4 * s + 1
    faces = []
    for i in range(s):
        j = (i + 1) % s
        # Shaft side (v0→v1)
        faces.append([i, j, s + j])
        faces.append([i, s + j, s + i])
        # Taper side (v1→v2, radius expands)
        faces.append([s + i, s + j, 2 * s + j])
        faces.append([s + i, 2 * s + j, 2 * s + i])
        # Plug bore side (v2→v3, same radius)
        faces.append([2 * s + i, 2 * s + j, 3 * s + j])
        faces.append([2 * s + i, 3 * s + j, 3 * s + i])
        # Top cap (center → v3 ring)
        faces.append([ic_t, 3 * s + i, 3 * s + j])
        # Bottom cap (center → v0 ring, reversed)
        faces.append([ic_b, j, i])
    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
    trimesh.repair.fix_normals(mesh)
    return mesh


def generate_screw_cap(output_dir, segments=24):
    """
    Generate a press-fit cap for M2 countersunk screw holes.

    The cap is a truncated cone (annular ring) matching the countersink taper,
    with a needle-sized hole through the center for removal access.
    Built directly as a mesh (no boolean needed).
    Saved as cap.obj and cap.stl in output_dir.
    """
    r_top = (COUNTERSINK_TOP_DIA - CAP_CLEARANCE) / 2
    r_bot = (COUNTERSINK_BOT_DIA - CAP_CLEARANCE) / 2
    h = COUNTERSINK_DEPTH + PLUG_BORE_DEPTH  # taper + plug bore
    needle_r = CAP_NEEDLE_HOLE / 2

    # Build annular frustum mesh directly: outer taper + inner needle hole
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    s = segments

    # Cap profile (top to bottom): plug cylinder (r_top) then taper (r_top→r_bot)
    z_top = h / 2
    z_taper_top = h / 2 - PLUG_BORE_DEPTH   # where taper begins
    z_bot = -h / 2
    # Outer rings: top (plug), taper transition, bottom
    o_top = np.column_stack([r_top * cos_a, r_top * sin_a, np.full(s, z_top)])
    o_mid = np.column_stack([r_top * cos_a, r_top * sin_a, np.full(s, z_taper_top)])
    o_bot = np.column_stack([r_bot * cos_a, r_bot * sin_a, np.full(s, z_bot)])
    # Inner needle hole rings (constant radius)
    i_top = np.column_stack([needle_r * cos_a, needle_r * sin_a, np.full(s, z_top)])
    i_bot = np.column_stack([needle_r * cos_a, needle_r * sin_a, np.full(s, z_bot)])

    verts = np.vstack([o_top, o_mid, o_bot, i_top, i_bot])
    # o_top:0, o_mid:s, o_bot:2s, i_top:3s, i_bot:4s

    faces = []
    for i in range(s):
        j = (i + 1) % s
        # Plug cylinder side (o_top→o_mid)
        faces.append([i, j, s + j])
        faces.append([i, s + j, s + i])
        # Taper side (o_mid→o_bot)
        faces.append([s + i, s + j, 2 * s + j])
        faces.append([s + i, 2 * s + j, 2 * s + i])
        # Inner side (i_top→i_bot, reversed winding)
        faces.append([3 * s + i, 4 * s + i, 4 * s + j])
        faces.append([3 * s + i, 4 * s + j, 3 * s + j])
        # Top annular face (o_top→i_top)
        faces.append([i, 3 * s + i, 3 * s + j])
        faces.append([i, 3 * s + j, j])
        # Bottom annular face (o_bot→i_bot, reversed)
        faces.append([2 * s + i, 2 * s + j, 4 * s + j])
        faces.append([2 * s + i, 4 * s + j, 4 * s + i])

    verts = np.array(verts)
    faces = np.array(faces)

    out_dir = Path(output_dir)
    cap_obj = out_dir / "cap.obj"
    cap_stl = out_dir / "cap.stl"

    # Write OBJ
    with open(cap_obj, 'w') as f:
        f.write("# Screw hole cap (press-fit, M2 countersunk)\n")
        f.write(f"# Top dia: {r_top*2:.2f}mm, Bot dia: {r_bot*2:.2f}mm\n")
        f.write(f"# Height: {h:.1f}mm, Needle hole: {CAP_NEEDLE_HOLE:.1f}mm\n")
        f.write("o ScrewCap\n")
        for v in verts:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        f.write("\n")
        for face in faces:
            f.write("f " + " ".join(str(i + 1) for i in face) + "\n")

    # Write STL
    import struct
    with open(cap_stl, 'wb') as f:
        f.write(b"STL cap".ljust(80, b'\0'))
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

    print(f"  Cap: {cap_obj} ({len(verts)}v, {len(faces)}f)")
    print(f"    Top dia: {r_top*2:.1f}mm, Bot dia: {r_bot*2:.1f}mm, "
          f"Height: {h:.1f}mm, Needle: {CAP_NEEDLE_HOLE:.1f}mm")
    return str(cap_obj), str(cap_stl)


def subtract_screw_holes(solid_verts, solid_faces, hole_positions, normal, pan_thickness,
                         hole_diameter=SCREW_HOLE_DIAMETER,
                         countersink_top=COUNTERSINK_TOP_DIA,
                         countersink_depth=COUNTERSINK_DEPTH,
                         segments=SCREW_HOLE_SEGMENTS):
    """
    Boolean-subtract M2 through-holes with tapered countersinks from pad solid.

    Each hole has two stages:
    - Through-hole (hole_diameter) spanning the full pad thickness
    - Tapered countersink on the playing surface (cone from countersink_top
      down to hole_diameter over countersink_depth)

    Returns new (vertices, faces) arrays with proper holes cut.
    """
    # Triangulate quads for trimesh (trimesh needs triangles)
    tri_faces = []
    for face in solid_faces:
        if len(face) == 3:
            tri_faces.append(face)
        elif len(face) == 4:
            tri_faces.append([face[0], face[1], face[2]])
            tri_faces.append([face[0], face[2], face[3]])
        else:
            for k in range(1, len(face) - 1):
                tri_faces.append([face[0], face[k], face[k + 1]])

    pad_mesh = trimesh.Trimesh(vertices=solid_verts, faces=np.array(tri_faces),
                               process=True)
    trimesh.repair.fix_normals(pad_mesh)
    trimesh.repair.fill_holes(pad_mesh)

    # Build rotation to align Z-axis with the surface normal
    z_axis = np.array([0, 0, 1.0])
    n_hat = normal / np.linalg.norm(normal)

    v = np.cross(z_axis, n_hat)
    c = np.dot(z_axis, n_hat)
    if np.linalg.norm(v) < 1e-10:
        rot_matrix = np.eye(3) if c > 0 else np.diag([1, -1, -1])
    else:
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])
        rot_matrix = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s * s))

    # Through-hole: cylinder extends through full pad thickness with overshoot
    overshoot = 1.0  # mm extra on each side for clean boolean cut
    cyl_height = pan_thickness + 2 * overshoot

    def _do_boolean(mesh, tool, label=""):
        """Boolean subtract with repair and engine fallback."""
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_winding(mesh)
        if not mesh.is_volume:
            mesh.merge_vertices(merge_tex=True, merge_norm=True)
            trimesh.repair.fix_normals(mesh)
            trimesh.repair.fill_holes(mesh)
        for engine in ('manifold', 'blender'):
            try:
                result = mesh.difference(tool, engine=engine)
                return result
            except Exception:
                continue
        print(f"    WARNING: boolean failed for {label}")
        return mesh

    for i, pos in enumerate(hole_positions):
        # Single compound tool: shaft through-hole + tapered countersink
        tool = _create_screw_tool(
            shaft_r=hole_diameter / 2,
            head_r=countersink_top / 2,
            total_h=cyl_height,
            taper_h=countersink_depth,
            segments=segments)

        # Centre the tool so its top aligns with the playing surface
        # (with overshoot above)
        centre = pos - n_hat * (pan_thickness / 2.0)

        T = np.eye(4)
        T[:3, :3] = rot_matrix
        T[:3, 3] = centre
        tool.apply_transform(T)

        pad_mesh = _do_boolean(pad_mesh, tool, f"screw {i}")

    return np.array(pad_mesh.vertices), [list(f) for f in pad_mesh.faces]


def _point_in_polygon_2d(px, py, poly_x, poly_y):
    """Ray-casting point-in-polygon test (2D)."""
    n = len(poly_x)
    inside = False
    j = n - 1
    for i in range(n):
        yi, yj = poly_y[i], poly_y[j]
        xi, xj = poly_x[i], poly_x[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _boss_contained_in_boundary(cx, cy, boss_radius, poly_x, poly_y):
    """Check if a boss circle is wholly inside the boundary polygon."""
    # Check center is inside
    if not _point_in_polygon_2d(cx, cy, poly_x, poly_y):
        return False
    # Check minimum distance from center to any polygon edge >= boss_radius
    n = len(poly_x)
    for i in range(n):
        j = (i + 1) % n
        # Distance from point (cx,cy) to line segment (xi,yi)-(xj,yj)
        ax, ay = poly_x[j] - poly_x[i], poly_y[j] - poly_y[i]
        bx, by = cx - poly_x[i], cy - poly_y[i]
        seg_len_sq = ax * ax + ay * ay
        if seg_len_sq < 1e-12:
            dist_sq = bx * bx + by * by
        else:
            t = max(0.0, min(1.0, (bx * ax + by * ay) / seg_len_sq))
            dx = bx - t * ax
            dy = by - t * ay
            dist_sq = dx * dx + dy * dy
        if dist_sq < boss_radius * boss_radius:
            return False
    return True


def compute_hole_positions(pan_verts, pan_faces, centroid, normal,
                           count=SCREW_HOLE_COUNT, inset=SCREW_HOLE_INSET,
                           mount_center=None, mount_clearance=0.0,
                           hw_mask_2d=None, tangent_xl=None, tangent_yl=None):
    """
    Compute screw hole positions equally spaced around the pad boundary.

    Algorithm:
    1. Extract boundary loop of pan faces
    2. Compute cumulative arc-length along the boundary
    3. Pick `count` points at equal arc-length intervals
    4. Step each point inward by `inset` mm toward the centroid
    5. Reject any point within `mount_clearance` of `mount_center`
       OR inside `hw_mask_2d` (symmetric hardware mask polygon in tangent plane)

    Returns list of positions in 3D world coordinates (on the surface).
    """
    z_local = normal / np.linalg.norm(normal)

    # Build orthonormal basis on tangent plane
    if abs(z_local[0]) < 0.9:
        x_local = np.cross(z_local, np.array([1, 0, 0]))
    else:
        x_local = np.cross(z_local, np.array([0, 1, 0]))
    x_local = x_local / np.linalg.norm(x_local)
    y_local = np.cross(z_local, x_local)

    # Get boundary loop
    be = find_boundary_edges(pan_faces)
    loops = find_all_boundary_loops(be)
    if not loops:
        # Fallback: use bounding box corners
        rel_pan = pan_verts - centroid
        pan_lx = rel_pan @ x_local
        pan_ly = rel_pan @ y_local
        corners_local = [
            (pan_lx.min() + inset, pan_ly.min() + inset),
            (pan_lx.max() - inset, pan_ly.min() + inset),
            (pan_lx.max() - inset, pan_ly.max() - inset),
            (pan_lx.min() + inset, pan_ly.max() - inset),
        ]
        return [centroid + cx * x_local + cy * y_local for cx, cy in corners_local]

    # Use the longest loop (main outer boundary)
    boundary_vi = max(loops, key=len)
    boundary_pts = pan_verts[boundary_vi]
    n_pts = len(boundary_pts)

    # Compute cumulative arc-length along boundary
    seg_lengths = np.linalg.norm(np.diff(boundary_pts, axis=0, append=boundary_pts[:1]), axis=1)
    cum_length = np.concatenate([[0], np.cumsum(seg_lengths[:-1])])
    total_length = seg_lengths.sum()

    # Oversample candidates to allow filtering near mount
    n_candidates = count * 4 if mount_clearance > 0 else count
    candidates = []
    for k in range(n_candidates):
        target = (k / n_candidates) * total_length
        idx = np.searchsorted(cum_length, target, side='right') - 1
        idx = max(0, min(idx, n_pts - 1))
        seg_start = cum_length[idx]
        seg_len = seg_lengths[idx]
        if seg_len < 1e-8:
            pt = boundary_pts[idx]
        else:
            t = (target - seg_start) / seg_len
            pt = boundary_pts[idx] * (1 - t) + boundary_pts[(idx + 1) % n_pts] * t
        # Step inward toward centroid
        direction = centroid - pt
        dist = np.linalg.norm(direction)
        if dist > 1e-6:
            pt = pt + (direction / dist) * inset
        candidates.append(pt)

    # Filter out candidates too close to the mount center
    if mount_center is not None and mount_clearance > 0:
        candidates = [p for p in candidates
                      if np.linalg.norm(p - mount_center) >= mount_clearance]

    # Filter out candidates inside the symmetric hardware mask
    if hw_mask_2d is not None and tangent_xl is not None:
        from shapely.geometry import Point as _HwPt
        filtered = []
        for p in candidates:
            rel = p - centroid
            px, py = float(rel @ tangent_xl), float(rel @ tangent_yl)
            if not hw_mask_2d.contains(_HwPt(px, py)):
                filtered.append(p)
        n_rejected = len(candidates) - len(filtered)
        if n_rejected > 0:
            print(f"  Rejected {n_rejected} holes in hardware mask zone")
        candidates = filtered

    # Select `count` candidates with maximum separation between them.
    # Greedy: pick first, then always pick the candidate furthest from
    # all already-selected holes, enforcing a minimum distance.
    if len(candidates) <= count:
        return candidates
    if not candidates:
        return []

    min_hole_sep = 8.0  # minimum mm between any two screw holes

    selected = [candidates[0]]
    for _ in range(count - 1):
        best_pt = None
        best_min_dist = -1
        for c in candidates:
            min_d = min(np.linalg.norm(c - s) for s in selected)
            if min_d >= min_hole_sep and min_d > best_min_dist:
                best_min_dist = min_d
                best_pt = c
        if best_pt is not None:
            selected.append(best_pt)
        else:
            break

    # If we didn't get enough holes, retry with reduced separation
    if len(selected) < count:
        for reduced_sep in [6.0, 4.0]:
            selected = [candidates[0]]
            for _ in range(count - 1):
                best_pt = None
                best_min_dist = -1
                for c in candidates:
                    min_d = min(np.linalg.norm(c - s) for s in selected)
                    if min_d >= reduced_sep and min_d > best_min_dist:
                        best_min_dist = min_d
                        best_pt = c
                if best_pt is not None:
                    selected.append(best_pt)
                else:
                    break
            if len(selected) >= count:
                print(f"  Found {count} holes with reduced separation ({reduced_sep}mm)")
                break

    return selected


def generate_notepad(note_index, obj_path, output_dir,
                     pan_thickness=PAN_THICKNESS,
                     grove_depth=GROVE_DEPTH,
                     grove_protrusion=GROVE_PROTRUSION,
                     groove_spread=1.0,
                     clone_from=None):
    """Generate a printable note pad for the given note index.

    groove_spread: for inner ring pads, scale factor for pad boundary extension.
        1.0 = original, >1.0 = extend outward.
    clone_from: if set, a note index (e.g. 'I1') whose geometry will be
        cloned, rotated and translated to this pad's position.
    """

    if note_index not in NOTE_BY_INDEX:
        print(f"Error: Unknown note index '{note_index}'")
        print(f"Valid indices: {sorted(NOTE_BY_INDEX.keys())}")
        return None

    note_info = NOTE_BY_INDEX[note_index]
    grove_obj = note_info['grove_object']
    pan_obj = note_info['pan_object']
    note_name = note_info['note']
    octave = note_info['octave']
    ring = note_info['ring']

    # Init per-pad state (prevent stale data from previous iterations)
    _inner_extended_ring = None
    _flange_solid_v = None
    _flange_solid_f = None
    _groove_attached = False

    print(f"\n{'='*60}")
    print(f"Generating Note Pad: {note_index} ({note_name}{octave}) - {ring} ring")
    print(f"{'='*60}")
    print(f"  Grove object: {grove_obj}")
    print(f"  Pan object: {pan_obj}")

    # Parse OBJ file
    print(f"\nParsing {obj_path}...")
    objects, all_vertices = parse_obj_file(obj_path)

    # Extract target pad surface and grove
    print(f"Extracting pan surface ({pan_obj})...")
    target_pan_verts, pan_faces = extract_object_mesh(objects, pan_obj, all_vertices)
    print(f"  Pan: {len(target_pan_verts)} vertices, {len(pan_faces)} faces")

    print(f"Extracting grove ({grove_obj})...")
    grove_verts, grove_faces = extract_object_mesh(objects, grove_obj, all_vertices)
    print(f"  Grove: {len(grove_verts)} vertices, {len(grove_faces)} faces")

    if clone_from is not None and clone_from in NOTE_BY_INDEX:
        # Clone geometry from source pad, transform to match target position.
        src_info = NOTE_BY_INDEX[clone_from]
        src_pan_verts, src_pan_faces = extract_object_mesh(
            objects, src_info['pan_object'], all_vertices)
        src_grove_verts, src_grove_faces = extract_object_mesh(
            objects, src_info['grove_object'], all_vertices)
        print(f"  ** CLONING from {clone_from} ({len(src_pan_verts)} verts)")

        src_centroid = src_pan_verts.mean(axis=0)
        tgt_centroid = target_pan_verts.mean(axis=0)

        if ring == 'outer':
            # Outer ring: rotate around the drum axis through an optimized
            # center that minimizes angular spacing and radius variance for
            # the 12 outer pads (not the raw Pan centroid which is off-center).
            import json as _cj
            # Optimized rotation center (in raw mm coords)
            _drum_center = np.array([0.2240, 798.5343, 4.0704])
            _R_level = compute_leveling_rotation(obj_path)
            drum_axis = _R_level.T @ np.array([0, 1, 0])  # drum axis in source coords
            drum_axis /= np.linalg.norm(drum_axis)

            # Project source/target centroids to plane perpendicular to drum axis
            src_rel = src_centroid - _drum_center
            tgt_rel = tgt_centroid - _drum_center
            # Remove drum-axis component
            src_flat = src_rel - np.dot(src_rel, drum_axis) * drum_axis
            tgt_flat = tgt_rel - np.dot(tgt_rel, drum_axis) * drum_axis
            # Angle between them around drum axis
            src_flat_n = src_flat / np.linalg.norm(src_flat)
            tgt_flat_n = tgt_flat / np.linalg.norm(tgt_flat)
            cos_a = np.clip(np.dot(src_flat_n, tgt_flat_n), -1, 1)
            sin_a = np.dot(np.cross(src_flat_n, tgt_flat_n), drum_axis)
            rot_angle_raw = math.atan2(sin_a, cos_a)
            # Snap to nearest 30° (outer pads are evenly spaced at 30°)
            rot_angle = round(rot_angle_raw / math.radians(30)) * math.radians(30)

            # Rodrigues rotation around drum_axis
            c_r, s_r = math.cos(rot_angle), math.sin(rot_angle)
            K = np.array([[0, -drum_axis[2], drum_axis[1]],
                          [drum_axis[2], 0, -drum_axis[0]],
                          [-drum_axis[1], drum_axis[0], 0]])
            R_drum = np.eye(3) + K * s_r + K @ K * (1 - c_r)

            pan_verts = (R_drum @ (src_pan_verts - _drum_center).T).T + _drum_center
            grove_verts = (R_drum @ (src_grove_verts - _drum_center).T).T + _drum_center
            pan_faces = src_pan_faces
            grove_faces = src_grove_faces
            print(f"  Outer clone: drum-axis rotation {math.degrees(rot_angle):.1f}°")
            # Shave groove edges inward by a small tolerance for printing fit
            GROOVE_CLONE_TOLERANCE = 0.3  # mm
            if len(grove_verts) >= 3:
                grove_centroid = grove_verts.mean(axis=0)
                for gi in range(len(grove_verts)):
                    to_center = grove_centroid - grove_verts[gi]
                    dist = np.linalg.norm(to_center)
                    if dist > 0.1:
                        grove_verts[gi] += to_center / dist * GROOVE_CLONE_TOLERANCE
                print(f"  Groove edges shaved {GROOVE_CLONE_TOLERANCE}mm inward for printing tolerance")
        else:
            # Inner/central: use normal alignment + long-axis twist + bbox scale
            src_normal = compute_surface_normal(src_pan_verts, src_pan_faces)
            tgt_normal = compute_surface_normal(target_pan_verts, pan_faces)
            tgt_bbox = target_pan_verts.max(axis=0) - target_pan_verts.min(axis=0)

            def rotation_between(a, b):
                a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
                v = np.cross(a, b)
                c = np.dot(a, b)
                if np.linalg.norm(v) < 1e-10:
                    return np.eye(3) if c > 0 else np.diag([1, -1, -1])
                s = np.linalg.norm(v)
                kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s * s))

            R_normal = rotation_between(src_normal, tgt_normal)

            def compute_long_axis(verts, normal, centroid):
                n = normal / np.linalg.norm(normal)
                if abs(n[0]) < 0.9: xl = np.cross(n, [1,0,0])
                else: xl = np.cross(n, [0,1,0])
                xl /= np.linalg.norm(xl)
                yl = np.cross(n, xl)
                rel = verts - centroid
                lx, ly = rel @ xl, rel @ yl
                cov = np.cov(np.column_stack([lx, ly]).T)
                vals, vecs = np.linalg.eigh(cov)
                long_2d = vecs[:, np.argmax(vals)]
                return long_2d[0] * xl + long_2d[1] * yl

            src_long = compute_long_axis(src_pan_verts, src_normal, src_centroid)
            tgt_long = compute_long_axis(target_pan_verts, tgt_normal, tgt_centroid)
            src_long_rotated = R_normal @ src_long
            tgt_n = tgt_normal / np.linalg.norm(tgt_normal)
            src_proj = src_long_rotated - np.dot(src_long_rotated, tgt_n) * tgt_n
            tgt_proj = tgt_long - np.dot(tgt_long, tgt_n) * tgt_n
            src_proj /= np.linalg.norm(src_proj)
            tgt_proj /= np.linalg.norm(tgt_proj)
            cos_a = np.clip(np.dot(src_proj, tgt_proj), -1, 1)
            sin_a = np.dot(np.cross(src_proj, tgt_proj), tgt_n)
            angle = math.atan2(sin_a, cos_a)
            c_a, s_a = math.cos(angle), math.sin(angle)
            kmat = np.array([[0, -tgt_n[2], tgt_n[1]],
                             [tgt_n[2], 0, -tgt_n[0]],
                             [-tgt_n[1], tgt_n[0], 0]])
            R_twist = np.eye(3) + kmat * s_a + kmat @ kmat * (1 - c_a)
            R = R_twist @ R_normal

            rotated_pan = (R @ (src_pan_verts - src_centroid).T).T
            rotated_grove = (R @ (src_grove_verts - src_centroid).T).T
            rot_bbox = rotated_pan.max(axis=0) - rotated_pan.min(axis=0)
            scale = np.ones(3)
            for ax in range(3):
                if rot_bbox[ax] > 1e-6:
                    scale[ax] = tgt_bbox[ax] / rot_bbox[ax]
            pan_verts = rotated_pan * scale + tgt_centroid
            pan_faces = src_pan_faces
            grove_verts = rotated_grove * scale + tgt_centroid
            grove_faces = src_grove_faces
            print(f"  Cloned, aligned (twist {math.degrees(angle):.1f}°), scaled ({scale[0]:.3f}, {scale[1]:.3f}, {scale[2]:.3f})")
    else:
        pan_verts = target_pan_verts

    # Select mount parameters based on ring
    if ring == 'outer':
        mount_inner_dia = MOUNT_INNER_DIAMETER
        mount_depth = MOUNT_DEPTH
        mount_wall = MOUNT_WALL_THICKNESS
        mount_thread_pitch = MOUNT_THREAD_PITCH
        mount_thread_depth = MOUNT_THREAD_DEPTH
        mount_notch_width = MOUNT_NOTCH_WIDTH
        min_pad_size = MIN_PAD_SIZE
    else:
        # Central and inner rings: boss sunk into pad
        mount_inner_dia = CENTRAL_MOUNT_INNER_DIAMETER
        mount_depth = CENTRAL_MOUNT_DEPTH
        mount_wall = CENTRAL_MOUNT_WALL_THICKNESS
        mount_thread_pitch = CENTRAL_MOUNT_THREAD_PITCH
        mount_thread_depth = CENTRAL_MOUNT_THREAD_DEPTH
        mount_notch_width = CENTRAL_MOUNT_NOTCH_WIDTH
        min_pad_size = CENTRAL_MIN_PAD_SIZE
        # pad_thickness stays at PAN_THICKNESS (5mm) — boss protrudes below

    # Check if pad needs scaling to accommodate mounting cylinder.
    # For inner ring, scale uniformly (all axes) to avoid curvature distortion.
    x_extent = pan_verts[:, 0].max() - pan_verts[:, 0].min()
    z_extent = pan_verts[:, 2].max() - pan_verts[:, 2].min()
    min_extent = min(x_extent, z_extent)

    MAX_INNER_SCALE = 1.15  # cap inner pad scaling to avoid overlap
    if min_extent < min_pad_size:
        scale_factor = min_pad_size / min_extent
        if ring == 'inner' and scale_factor > MAX_INNER_SCALE:
            print(f"  ** Capping inner pad scale from {scale_factor:.3f}x to {MAX_INNER_SCALE}x")
            scale_factor = MAX_INNER_SCALE
        was_scaled = True
        if ring == 'inner':
            # Uniform 3D scale from centroid to preserve curvature
            center = pan_verts.mean(axis=0)
            pan_verts = center + (pan_verts - center) * scale_factor
            grove_center = grove_verts.mean(axis=0)
            grove_verts = grove_center + (grove_verts - grove_center) * scale_factor
        else:
            pan_verts, grove_verts, scale_factor, was_scaled = check_and_scale_pad(
                pan_verts, grove_verts, min_size=min_pad_size)
        print(f"  ** PAD SCALED by {scale_factor:.3f}x to fit mounting cylinder (min size: {min_pad_size}mm)")
    else:
        scale_factor = 1.0
        was_scaled = False

    # Nudge specific pads to fix overlaps (rotate around inner ring centroid)
    # I0 overlaps I4 — rotate I0 toward I1 by a small angle
    INNER_PAD_NUDGES = {
        'I0': -7.0,  # degrees — negative = toward I1 (clockwise in XZ)
    }
    if note_index in INNER_PAD_NUDGES and ring == 'inner':
        _nudge_deg = INNER_PAD_NUDGES[note_index]
        _nudge_rad = math.radians(_nudge_deg)
        # Compute inner ring centroid from all inner pad positions
        _inner_centers = []
        for _iidx in ['I0', 'I1', 'I2', 'I3', 'I4']:
            if _iidx in NOTE_BY_INDEX:
                _ipv, _ipf = extract_object_mesh(objects, NOTE_BY_INDEX[_iidx]['pan_object'], all_vertices)
                _inner_centers.append(_ipv.mean(axis=0))
        _ring_center = np.mean(_inner_centers, axis=0)
        # Rotate pad + groove around ring center in XZ plane
        _c, _s = math.cos(_nudge_rad), math.sin(_nudge_rad)
        for _verts in [pan_verts, grove_verts]:
            _rel = _verts - _ring_center
            _new_x = _rel[:, 0] * _c - _rel[:, 2] * _s
            _new_z = _rel[:, 0] * _s + _rel[:, 2] * _c
            _verts[:, 0] = _ring_center[0] + _new_x
            _verts[:, 2] = _ring_center[2] + _new_z
        print(f"  ** Nudged {note_index} by {_nudge_deg:.1f}° around inner ring center")

    # Compute surface normal from pan (the playing surface defines the orientation)
    pan_normal = compute_surface_normal(pan_verts, pan_faces)
    print(f"  Pan surface normal: ({pan_normal[0]:.4f}, {pan_normal[1]:.4f}, {pan_normal[2]:.4f})")

    # Shrink pad horizontally to ensure it fits in the groove pocket
    if PAD_SHRINK > 0:
        pad_center = pan_verts.mean(axis=0)
        n_hat_s = pan_normal / np.linalg.norm(pan_normal)
        for i in range(len(pan_verts)):
            to_v = pan_verts[i] - pad_center
            # Project onto tangent plane (remove normal component)
            planar = to_v - np.dot(to_v, n_hat_s) * n_hat_s
            r = np.linalg.norm(planar)
            if r > 0.1:
                # Shrink by PAD_SHRINK mm inward
                shrink_frac = max(0, (r - PAD_SHRINK) / r)
                pan_verts[i] = pad_center + np.dot(to_v, n_hat_s) * n_hat_s + planar * shrink_frac

    # Thicken pan surface (downward only)
    print(f"Thickening pan surface (down: {pan_thickness}mm)...")
    _groove_attached = False
    if ring == 'inner':
        # Inner pads: thicken the pad only (no groove attachment).
        pan_solid_verts, pan_solid_faces = thicken_surface(pan_verts, pan_faces,
                                                            thickness_down=pan_thickness,
                                                            thickness_up=0)
        print(f"  Pan solid: {len(pan_solid_verts)} vertices, {len(pan_solid_faces)} faces")
    elif ring == 'central':
        # Central pads: thicken pad, then also thicken groove and combine.
        # This extends the pad footprint into groove area so screw holes
        # have material underneath them.
        pan_solid_verts, pan_solid_faces = thicken_surface(pan_verts, pan_faces,
                                                            thickness_down=pan_thickness,
                                                            thickness_up=0)
        print(f"  Pan solid: {len(pan_solid_verts)} vertices, {len(pan_solid_faces)} faces")
        # Attach groove to central pad
        if len(grove_verts) >= 3 and len(grove_faces) >= 1:
            grove_solid_v, grove_solid_f = thicken_surface(grove_verts, grove_faces,
                                                            thickness_down=pan_thickness,
                                                            thickness_up=0)
            # Combine: offset groove face indices by pad vertex count
            n_pad_v = len(pan_solid_verts)
            pan_solid_verts = np.vstack([pan_solid_verts, grove_solid_v])
            pan_solid_faces += [[vi + n_pad_v for vi in f] for f in grove_solid_f]
            _groove_attached = True
            print(f"  + Groove attached: {len(grove_solid_v)}v, combined {len(pan_solid_verts)}v")
    else:
        # Outer ring: thicken pad, then attach groove (same approach as central)
        pan_solid_verts, pan_solid_faces = thicken_surface(pan_verts, pan_faces,
                                                            thickness_down=pan_thickness,
                                                            thickness_up=0)
        print(f"  Pan solid: {len(pan_solid_verts)} vertices, {len(pan_solid_faces)} faces")
        # Attach groove to outer pad
        if len(grove_verts) >= 3 and len(grove_faces) >= 1:
            grove_solid_v, grove_solid_f = thicken_surface(grove_verts, grove_faces,
                                                            thickness_down=pan_thickness,
                                                            thickness_up=0)
            n_pad_v = len(pan_solid_verts)
            pan_solid_verts = np.vstack([pan_solid_verts, grove_solid_v])
            pan_solid_faces += [[vi + n_pad_v for vi in f] for f in grove_solid_f]
            _groove_attached = True
            print(f"  + Groove attached: {len(grove_solid_v)}v, combined {len(pan_solid_verts)}v")
    print(f"  Pan boundary loops: {len(find_all_boundary_loops(find_boundary_edges(pan_faces)))}")

    # Compute interior centroid for pan (must lie within the solid volume)
    pan_interior_centroid, pan_surface_centroid = compute_interior_centroid(
        pan_verts, pan_faces, pan_normal, pan_thickness, 0)

    # For outer ring pads, use the boundary-loop centroid projected onto the
    # surface plane so that the mount position is consistent across all pads.
    # (O3/O9 have extra internal geometry that pulls the vertex mean off-center;
    #  the boundary loop outline is symmetric and stable.)
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
        pan_interior_centroid = pan_surface_centroid - pan_normal * (pan_thickness / 2.0)

    print(f"  Mount position: ({pan_surface_centroid[0]:.2f}, {pan_surface_centroid[1]:.2f}, {pan_surface_centroid[2]:.2f})")

    # For inner ring pads, extend the pad surface outward by adding a
    # contiguous flange ring around the boundary.  This creates a single
    # surface with no gap (unlike attaching a separate groove mesh).
    # groove_spread controls how far each boundary vertex is pushed out
    # radially (as a fraction of its distance from the centroid).
    if ring == 'inner':
        print(f"Extending inner pad boundary (spread={groove_spread:.2f}x)...")

        # Find boundary loop of the ORIGINAL (pre-thicken) pan surface
        be = find_boundary_edges(pan_faces)
        loops = find_all_boundary_loops(be)
        if loops:
            boundary_vi = max(loops, key=len)
            n_orig = len(pan_verts)

            # Decimate the boundary loop: merge vertices closer than 0.5mm
            # to avoid degenerate flange quads on dense boundaries.
            boundary_pts_raw = pan_verts[boundary_vi]
            boundary_vi_clean = [boundary_vi[0]]
            for k in range(1, len(boundary_vi)):
                if np.linalg.norm(boundary_pts_raw[k] - pan_verts[boundary_vi_clean[-1]]) >= 0.5:
                    boundary_vi_clean.append(boundary_vi[k])
            boundary_vi = np.array(boundary_vi_clean)
            boundary_pts = pan_verts[boundary_vi]
            n_bnd = len(boundary_vi)
            boundary_centroid = boundary_pts.mean(axis=0)
            print(f"  Boundary: {n_bnd} vertices (after cleanup)")

            # For each boundary vertex, compute outward extension using
            # the edge-perpendicular direction in the tangent plane.
            extended_pts = []
            for k in range(n_bnd):
                pt = boundary_pts[k]
                pt_prev = boundary_pts[(k - 1) % n_bnd]
                pt_next = boundary_pts[(k + 1) % n_bnd]

                edge_dir = pt_next - pt_prev
                outward = np.cross(pan_normal, edge_dir)
                to_center = boundary_centroid - pt
                if np.dot(outward, to_center) > 0:
                    outward = -outward
                out_len = np.linalg.norm(outward)
                if out_len > 1e-8:
                    outward = outward / out_len
                else:
                    outward = pt - boundary_centroid
                    outward = outward - np.dot(outward, pan_normal) * pan_normal
                    ol = np.linalg.norm(outward)
                    outward = outward / ol if ol > 1e-8 else np.array([1, 0, 0])

                radial_dist = np.linalg.norm(pt - boundary_centroid)
                # Use percentage-based OR minimum extension.
                # Must be large enough for pin head (1.9mm r) + edge margin.
                min_ext = 6.5 if note_index == 'I4' else 6.0
                extension = max(radial_dist * (groove_spread - 1.0), min_ext)
                ext_pt = pt + outward * extension
                extended_pts.append(ext_pt)

            extended_pts = np.array(extended_pts)

            # Add new vertices for the flange ring. Connect with triangles.
            # The original pad surface stays untouched.
            ext_verts = np.vstack([pan_verts, extended_pts])
            ext_faces = list(pan_faces)

            for k in range(n_bnd):
                vi_curr = boundary_vi[k]
                vi_next = boundary_vi[(k + 1) % n_bnd]
                ext_curr = n_orig + k
                ext_next = n_orig + (k + 1) % n_bnd
                ext_faces.append([vi_curr, vi_next, ext_next])
                ext_faces.append([vi_curr, ext_next, ext_curr])

            print(f"  Flange: {n_bnd} new vertices, pad surface untouched")
            print(f"  Extended surface: {len(ext_verts)} verts, {len(ext_faces)} faces")

            # Save extended ring for screw hole placement
            _inner_extended_ring = extended_pts

            # Thicken the flange strip into a proper manifold solid,
            # then fix the bottom vertices to use the pad surface normal
            # (avoids divergent per-vertex normals that double thickness).
            flange_faces_only = ext_faces[len(pan_faces):]
            _flange_solid_v, _flange_solid_f = thicken_surface(
                ext_verts, flange_faces_only,
                thickness_down=pan_thickness, thickness_up=0)
            # Override bottom vertex positions: uniform pan_normal direction
            n_fv = len(ext_verts)
            for i in range(n_fv):
                _flange_solid_v[i + n_fv] = _flange_solid_v[i] - pan_normal * pan_thickness
            print(f"  Flange solid: {len(_flange_solid_v)} verts (appended after screw holes)")
        else:
            print(f"  WARNING: no boundary loop found, skipping extension")

    # Combined note pad properties (use pan's normal and centroid as reference)
    notepad_normal = pan_normal
    notepad_centroid = pan_interior_centroid

    # Compute pad long axis (PCA on tangent-plane projection) for plate orientation
    n_hat = notepad_normal / np.linalg.norm(notepad_normal)
    if abs(n_hat[0]) < 0.9:
        _xl = np.cross(n_hat, np.array([1, 0, 0]))
    else:
        _xl = np.cross(n_hat, np.array([0, 1, 0]))
    _xl /= np.linalg.norm(_xl)
    _yl = np.cross(n_hat, _xl)
    _rel = pan_verts - pan_surface_centroid
    _lx, _ly = _rel @ _xl, _rel @ _yl
    _cov = np.cov(np.column_stack([_lx, _ly]).T)
    _vals, _vecs = np.linalg.eigh(_cov)
    _long_2d = _vecs[:, np.argmax(_vals)]
    pad_long_axis = _long_2d[0] * _xl + _long_2d[1] * _yl
    pad_long_axis /= np.linalg.norm(pad_long_axis)

    # Generate mounting cylinder (sized per ring)
    print(f"Generating mounting cylinder ({ring} ring)...")
    print(f"  Inner diameter: {mount_inner_dia}mm, Depth: {mount_depth}mm")
    print(f"  Thread pitch: {mount_thread_pitch}mm, Notch width: {mount_notch_width}mm")

    # Central/inner rings: internal push-fit threads (cap pushes into boss)
    # Outer ring: external threads (mount base screws over boss)
    _boss_threads_inside = ring in ('central', 'inner')
    cylinder_verts, cylinder_faces = generate_threaded_mount_cylinder(
        inner_diameter=mount_inner_dia,
        depth=mount_depth,
        wall_thickness=mount_wall,
        thread_pitch=mount_thread_pitch,
        thread_depth=mount_thread_depth,
        notch_width=mount_notch_width,
        threads_inside=_boss_threads_inside
    )
    _thread_loc = "internal (push-cap)" if _boss_threads_inside else "external (mount base)"
    print(f"  Cylinder: {len(cylinder_verts)} vertices, {len(cylinder_faces)} faces ({_thread_loc})")

    # Position cylinder at pan surface centroid, oriented along normal.
    # For central/inner: sink the boss into the pad so its top is
    # SENSOR_SKIN mm below the playing surface (sensor close to surface).
    # For outer: cylinder top at the pad bottom surface (hangs below).
    if ring in ('central', 'inner'):
        # Boss top sunk into pad, SENSOR_SKIN below playing surface
        boss_origin = pan_surface_centroid - notepad_normal * SENSOR_SKIN
    else:
        boss_origin = pan_surface_centroid

    cylinder_verts_transformed = transform_cylinder_to_normal(
        cylinder_verts, boss_origin, notepad_normal)

    # For central/inner rings, rotate the boss cylinder about the normal so
    # that its wire notch (at local angle=0) aligns with the pad's long axis.
    # This ensures the notch lines up with the push-cap slit.
    if ring in ('central', 'inner'):
        # Find current local X direction after transform
        local_x = transform_cylinder_to_normal(
            np.array([[1.0, 0, 0]]), np.zeros(3), notepad_normal)[0]
        n_hat = notepad_normal / np.linalg.norm(notepad_normal)
        local_x = local_x - np.dot(local_x, n_hat) * n_hat
        lx_len = np.linalg.norm(local_x)
        if lx_len > 1e-6:
            local_x = local_x / lx_len
            # Target: pad long axis
            target_x = pad_long_axis - np.dot(pad_long_axis, n_hat) * n_hat
            target_x = target_x / np.linalg.norm(target_x)
            cos_a = np.clip(np.dot(local_x, target_x), -1, 1)
            sin_a = np.dot(np.cross(local_x, target_x), n_hat)
            twist = math.atan2(sin_a, cos_a)
            # Apply Rodrigues rotation about n_hat
            c_t, s_t = math.cos(twist), math.sin(twist)
            centered = cylinder_verts_transformed - pan_surface_centroid
            cylinder_verts_transformed = (
                centered * c_t +
                np.cross(n_hat, centered) * s_t +
                n_hat * np.dot(centered, n_hat).reshape(-1, 1) * (1 - c_t)
            ) + pan_surface_centroid
            print(f"  Boss notch aligned with pad long axis (twist {math.degrees(twist):.1f}°)")

    # Check if cylinder protrudes through curved surface and lower if needed
    # (only relevant for outer ring where boss sits at the surface)
    if ring == 'outer':
        cylinder_outer_radius = mount_inner_dia / 2 + mount_wall + mount_thread_depth
        surface_offset = compute_cylinder_surface_offset(
            pan_verts, pan_surface_centroid, notepad_normal, cylinder_outer_radius)
        if surface_offset < 0:
            cylinder_verts_transformed = cylinder_verts_transformed + notepad_normal * surface_offset
            print(f"  Cylinder lowered by {-surface_offset:.2f}mm to avoid surface protrusion")

    # Generate M2 through-holes for all rings
    print(f"Generating M2 through-holes (boolean subtraction)...")
    print(f"  Hole diameter: {SCREW_HOLE_DIAMETER}mm, Through full {pan_thickness}mm thickness")

    # Compute symmetric hardware mask in tangent plane for ALL rings.
    # This matches the yellow zones on the maps — screw holes must avoid this area.
    from scipy.spatial import ConvexHull as _ScrewCH
    from shapely.geometry import Polygon as _ScrewPoly, Point as _ScrewPt
    hw_radius = mount_inner_dia / 2 + mount_wall
    _n_hat_hw = notepad_normal / np.linalg.norm(notepad_normal)
    if abs(_n_hat_hw[0]) < 0.9:
        _xl_hw = np.cross(_n_hat_hw, [1, 0, 0])
    else:
        _xl_hw = np.cross(_n_hat_hw, [0, 1, 0])
    _xl_hw /= np.linalg.norm(_xl_hw)
    _yl_hw = np.cross(_n_hat_hw, _xl_hw)
    _cyl_rel = cylinder_verts_transformed - pan_surface_centroid
    _cyl_2d = np.column_stack([_cyl_rel @ _xl_hw, _cyl_rel @ _yl_hw])
    _hw_buffer = 1.5 if ring == 'inner' else 3.0  # buffer distance
    try:
        _ch = _ScrewCH(_cyl_2d)
        _hw_poly = _ScrewPoly(_cyl_2d[_ch.vertices])
        _long_2d = np.array([pad_long_axis @ _xl_hw, pad_long_axis @ _yl_hw])
        _long_2d /= np.linalg.norm(_long_2d)
        _coords = np.array(_hw_poly.exterior.coords)
        _proj = np.outer(_coords @ _long_2d, _long_2d)
        _reflected = 2 * _proj - _coords
        _mirror = _ScrewPoly(_reflected)
        _hw_poly = _hw_poly.union(_mirror)
        if _hw_poly.geom_type == 'MultiPolygon':
            _hw_poly = _hw_poly.convex_hull
        _hw_mask = _hw_poly.buffer(_hw_buffer)
        print(f"  Symmetric hardware mask: area={_hw_mask.area:.0f}mm² (buffered {_hw_buffer}mm)")
    except Exception:
        _hw_mask = _ScrewPt(0, 0).buffer(hw_radius + _hw_buffer)
        print(f"  Hardware mask: circular fallback r={hw_radius + _hw_buffer:.1f}mm")

    if ring == 'inner' or ring == 'central':
        # Grid-based screw hole placement: sample candidate points inside
        # the pad, reject any too close to the edge or the hardware zone,
        # then greedily pick well-separated positions.
        if ring == 'inner':
            MIN_EDGE_DIST = 5.0   # mm from pad boundary
            MIN_HW_DIST = 2.5     # mm from nearest hardware edge
            MIN_HOLE_SEP = 10.0   # mm between holes
            MAX_HOLES = 2         # 2 screws for inner pads
        else:
            MIN_EDGE_DIST = 6.0   # mm from pad boundary
            MIN_HW_DIST = 4.0     # mm from nearest hardware edge
            MIN_HOLE_SEP = 10.0   # mm between holes
            MAX_HOLES = 4

        # Build tangent-plane basis
        n_hat = notepad_normal / np.linalg.norm(notepad_normal)
        if abs(n_hat[0]) < 0.9:
            _xl = np.cross(n_hat, [1, 0, 0])
        else:
            _xl = np.cross(n_hat, [0, 1, 0])
        _xl /= np.linalg.norm(_xl)
        _yl = np.cross(n_hat, _xl)

        # Get boundary in 2D tangent plane
        # For inner pads with flange, use the extended ring as the boundary
        # For central pads with groove attached, use the groove outer boundary
        if ring == 'inner' and _inner_extended_ring is not None:
            bpts = _inner_extended_ring
            _have_boundary = True
        elif ring == 'central' and _groove_attached:
            # Use pad boundary but with relaxed edge distance — groove provides
            # extra material beyond the pad edge for screw hole support
            be = find_boundary_edges(pan_faces)
            loops = find_all_boundary_loops(be)
            _have_boundary = bool(loops)
            if _have_boundary:
                bvi = max(loops, key=len)
                bpts = pan_verts[bvi]
                MIN_EDGE_DIST = 5.0   # relaxed — groove extends beyond pad
                print(f"  Using pad boundary with relaxed edge dist ({MIN_EDGE_DIST}mm, groove attached)")
        else:
            be = find_boundary_edges(pan_faces)
            loops = find_all_boundary_loops(be)
            _have_boundary = bool(loops)
            if _have_boundary:
                bvi = max(loops, key=len)
                bpts = pan_verts[bvi]
        if _have_boundary:
            rel_b = bpts - pan_surface_centroid
            bx_2d = rel_b @ _xl
            by_2d = rel_b @ _yl

            # Sample grid of candidate points in tangent plane
            margin = 2.0
            x_min, x_max = bx_2d.min() + margin, bx_2d.max() - margin
            y_min, y_max = by_2d.min() + margin, by_2d.max() - margin
            step = 1.0  # 1mm grid

            candidates = []
            for gx in np.arange(x_min, x_max, step):
                for gy in np.arange(y_min, y_max, step):
                    # Check inside boundary (point-in-polygon)
                    if not _point_in_polygon_2d(gx, gy, bx_2d, by_2d):
                        continue
                    # Check distance from boundary edges
                    n_bnd = len(bx_2d)
                    edge_ok = True
                    for i in range(n_bnd):
                        j = (i + 1) % n_bnd
                        ax, ay = bx_2d[j] - bx_2d[i], by_2d[j] - by_2d[i]
                        bx_, by_ = gx - bx_2d[i], gy - by_2d[i]
                        seg_len_sq = ax * ax + ay * ay
                        if seg_len_sq < 1e-12:
                            d_sq = bx_ * bx_ + by_ * by_
                        else:
                            t = max(0, min(1, (bx_ * ax + by_ * ay) / seg_len_sq))
                            dx, dy = bx_ - t * ax, by_ - t * ay
                            d_sq = dx * dx + dy * dy
                        if d_sq < MIN_EDGE_DIST ** 2:
                            edge_ok = False
                            break
                    if not edge_ok:
                        continue
                    # Check against symmetric hardware mask polygon
                    if _hw_mask.contains(_ScrewPt(gx, gy)):
                        continue
                    hw_dist = math.sqrt(gx * gx + gy * gy)
                    candidates.append((gx, gy, hw_dist))

            if ring == 'inner' and len(candidates) >= 2:
                # Inner pads: place 2 holes at OPPOSITE ENDS of long axis
                cand_arr = np.array([(c[0], c[1]) for c in candidates])
                cand_mean = cand_arr.mean(axis=0)
                cand_centered = cand_arr - cand_mean
                cov = cand_centered.T @ cand_centered
                eigvals, eigvecs = np.linalg.eigh(cov)
                long_ax = eigvecs[:, 1]  # largest eigenvalue
                projections = cand_centered @ long_ax
                idx_pos = int(np.argmax(projections))
                idx_neg = int(np.argmin(projections))
                hole_positions = []
                for ci in [idx_pos, idx_neg]:
                    gx, gy = cand_arr[ci]
                    pt_3d = pan_surface_centroid + gx * _xl + gy * _yl
                    hole_positions.append(pt_3d)
                sep = np.linalg.norm(hole_positions[0] - hole_positions[1])
                if sep < 5.0:
                    # Holes too close — only keep the one furthest from hardware
                    dists = [np.linalg.norm(h - pan_surface_centroid) for h in hole_positions]
                    hole_positions = [hole_positions[int(np.argmax(dists))]]
                    print(f"  Inner ring: 1 hole only (sep={sep:.1f}mm too small, {len(candidates)} candidates)")
                else:
                    print(f"  Inner ring: 2 holes at opposite ends (sep={sep:.1f}mm, {len(candidates)} candidates)")
            elif ring == 'inner' and len(candidates) < 2:
                # Not enough candidates — relax constraints and retry
                print(f"  Inner ring: only {len(candidates)} candidates, relaxing constraints...")
                relaxed_candidates = []
                for gx in np.arange(x_min - 1, x_max + 1, step):
                    for gy in np.arange(y_min - 1, y_max + 1, step):
                        if not _point_in_polygon_2d(gx, gy, bx_2d, by_2d):
                            continue
                        # Relaxed edge distance (1.5mm instead of 2.2mm)
                        n_bnd = len(bx_2d)
                        edge_ok = True
                        for i in range(n_bnd):
                            j_idx = (i + 1) % n_bnd
                            ax, ay = bx_2d[j_idx] - bx_2d[i], by_2d[j_idx] - by_2d[i]
                            bx_, by_ = gx - bx_2d[i], gy - by_2d[i]
                            seg_len_sq = ax * ax + ay * ay
                            if seg_len_sq < 1e-12:
                                d_sq = bx_ * bx_ + by_ * by_
                            else:
                                t = max(0, min(1, (bx_ * ax + by_ * ay) / seg_len_sq))
                                dx, dy = bx_ - t * ax, by_ - t * ay
                                d_sq = dx * dx + dy * dy
                            if d_sq < 1.5 ** 2:  # relaxed
                                edge_ok = False
                                break
                        if not edge_ok:
                            continue
                        if _hw_mask.contains(_ScrewPt(gx, gy)):
                            continue
                        hw_dist = math.sqrt(gx * gx + gy * gy)
                        relaxed_candidates.append((gx, gy, hw_dist))
                if len(relaxed_candidates) >= 2:
                    cand_arr = np.array([(c[0], c[1]) for c in relaxed_candidates])
                    cand_mean = cand_arr.mean(axis=0)
                    cand_centered = cand_arr - cand_mean
                    cov = cand_centered.T @ cand_centered
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    long_ax = eigvecs[:, 1]
                    projections = cand_centered @ long_ax
                    idx_pos = int(np.argmax(projections))
                    idx_neg = int(np.argmin(projections))
                    hole_positions = []
                    for ci in [idx_pos, idx_neg]:
                        gx, gy = cand_arr[ci]
                        pt_3d = pan_surface_centroid + gx * _xl + gy * _yl
                        hole_positions.append(pt_3d)
                    sep = np.linalg.norm(hole_positions[0] - hole_positions[1])
                    print(f"  Inner ring: 2 holes (relaxed, sep={sep:.1f}mm, {len(relaxed_candidates)} candidates)")
                else:
                    # Last resort: use whatever candidates we have
                    hole_positions = []
                    for gx, gy, _ in (candidates + relaxed_candidates):
                        pt_3d = pan_surface_centroid + gx * _xl + gy * _yl
                        hole_positions.append(pt_3d)
                        if len(hole_positions) >= MAX_HOLES:
                            break
                    print(f"  Inner ring: {len(hole_positions)} holes (fallback)")
            else:
                # Central pads or fallback: greedy furthest-from-hardware
                candidates.sort(key=lambda c: -c[2])
                hole_positions = []
                for gx, gy, _ in candidates:
                    pt_3d = pan_surface_centroid + gx * _xl + gy * _yl
                    too_close = any(np.linalg.norm(pt_3d - h) < MIN_HOLE_SEP for h in hole_positions)
                    if not too_close:
                        hole_positions.append(pt_3d)
                    if len(hole_positions) >= MAX_HOLES:
                        break
                print(f"  {ring.title()} ring: {len(hole_positions)} holes placed (grid search, {len(candidates)} candidates)")
        elif not _have_boundary:
            hole_positions = []
            print(f"  {ring.title()} ring: no boundary found")
    else:
        # Outer ring: use symmetric hardware mask for filtering
        hole_positions = compute_hole_positions(
            pan_verts, pan_faces, pan_surface_centroid, notepad_normal,
            inset=SCREW_HOLE_INSET,
            mount_center=pan_surface_centroid, mount_clearance=0.0,
            hw_mask_2d=_hw_mask, tangent_xl=_xl_hw, tangent_yl=_yl_hw)

    if hole_positions:
        pre_vert_count = len(pan_solid_verts)
        pan_solid_verts, pan_solid_faces = subtract_screw_holes(
            pan_solid_verts, pan_solid_faces, hole_positions,
            notepad_normal, pan_thickness)

        # Check if booleans actually worked (vertex count should increase)
        if len(pan_solid_verts) <= pre_vert_count + 10:
            # Boolean likely failed — rebuild pad from boundary polygon
            print(f"  Boolean may have failed — rebuilding pad from boundary extrusion...")
            from shapely.geometry import Polygon as ShapelyPolygon
            n_hat = notepad_normal / np.linalg.norm(notepad_normal)
            if abs(n_hat[0]) < 0.9:
                _xl2 = np.cross(n_hat, [1, 0, 0])
            else:
                _xl2 = np.cross(n_hat, [0, 1, 0])
            _xl2 /= np.linalg.norm(_xl2)
            _yl2 = np.cross(n_hat, _xl2)

            # For inner pads, use the groove outer boundary (convex hull)
            # so the extrusion includes the groove area for screw holes.
            if ring == 'inner' and _inner_extended_ring is not None:
                bpts2 = _inner_extended_ring
            else:
                be2 = find_boundary_edges(pan_faces)
                loops2 = find_all_boundary_loops(be2)
                if loops2:
                    bvi2 = max(loops2, key=len)
                    bpts2 = pan_verts[bvi2]
                else:
                    bpts2 = None
            if bpts2 is not None:
                # Use boundary-loop centroid (not vertex mean) to avoid
                # displacement when internal geometry skews the mean (e.g. O9).
                centroid2 = bpts2.mean(axis=0)
                rel2 = bpts2 - centroid2
                bx2 = rel2 @ _xl2
                by2 = rel2 @ _yl2

                # For inner pads, groove boundary is a strip — use convex hull
                if ring == 'inner':
                    from scipy.spatial import ConvexHull as _CH2
                    _hull2 = _CH2(np.column_stack([bx2, by2]))
                    bx2 = bx2[_hull2.vertices]
                    by2 = by2[_hull2.vertices]

                poly = ShapelyPolygon(zip(bx2, by2))
                if not poly.is_valid:
                    poly = poly.buffer(0)

                if poly.is_valid and poly.area > 1:
                    extruded = trimesh.creation.extrude_polygon(poly, height=pan_thickness)

                    # Cut holes from the clean extruded mesh
                    for hp in hole_positions:
                        rel_hp = hp - centroid2
                        hx = np.dot(rel_hp, _xl2)
                        hy = np.dot(rel_hp, _yl2)

                        # Compound screw tool (shaft + tapered countersink)
                        tool = _create_screw_tool(
                            shaft_r=SCREW_HOLE_DIAMETER / 2,
                            head_r=COUNTERSINK_TOP_DIA / 2,
                            total_h=pan_thickness + 2,
                            taper_h=COUNTERSINK_DEPTH,
                            segments=16)
                        # Rotate 180° about X so countersink faces -Z
                        # (local -Z → world +n_hat = playing surface side)
                        R180 = np.eye(4)
                        R180[1, 1] = -1
                        R180[2, 2] = -1
                        tool.apply_transform(R180)
                        T = np.eye(4)
                        T[0, 3] = hx
                        T[1, 3] = hy
                        T[2, 3] = pan_thickness / 2
                        tool.apply_transform(T)
                        try:
                            extruded = extruded.difference(tool, engine='manifold')
                        except Exception:
                            pass

                    # Transform back to 3D world coords
                    # The extruded mesh is in local 2D+Z coords — transform to world
                    ext_v = np.array(extruded.vertices)
                    world_v = np.zeros_like(ext_v)
                    world_v = (centroid2 +
                               np.outer(ext_v[:, 0], _xl2) +
                               np.outer(ext_v[:, 1], _yl2) -
                               np.outer(ext_v[:, 2], n_hat))
                    pan_solid_verts = world_v
                    pan_solid_faces = [list(f) for f in extruded.faces]
                    print(f"  Rebuilt from extrusion: {len(pan_solid_verts)} verts, {len(pan_solid_faces)} faces")

        print(f"  Subtracted {len(hole_positions)} M2 through-holes from pan solid")

    # For central/inner pads, the boss is sunk into the pad body.
    if ring in ('central', 'inner'):
        print(f"  Boss sunk into {pan_thickness:.1f}mm thick pad")
        print(f"  Sensor at {SENSOR_SKIN}mm below playing surface")

    # Append flange solid if we have one (inner ring pads)
    # Also cut holes through the flange where needed
    if _flange_solid_v is not None:
        if hole_positions:
            print(f"  Cutting {len(hole_positions)} holes through flange...")
            _flange_solid_v, _flange_solid_f = subtract_screw_holes(
                _flange_solid_v, _flange_solid_f, hole_positions,
                notepad_normal, pan_thickness)
        n_pan = len(pan_solid_verts)
        pan_solid_verts = np.vstack([pan_solid_verts, _flange_solid_v])
        pan_solid_faces = pan_solid_faces + [[i + n_pan for i in f] for f in _flange_solid_f]
        print(f"  Appended flange: {len(pan_solid_verts)} verts, {len(pan_solid_faces)} faces")

    # Combine: pan (with holes + flange) + mount cylinder
    n_pan_solid_verts = len(pan_solid_verts)
    solid_verts = np.vstack([pan_solid_verts, cylinder_verts_transformed])
    solid_faces = pan_solid_faces + [[idx + n_pan_solid_verts for idx in face] for face in cylinder_faces]
    print(f"  Combined pan + cylinder: {len(solid_verts)} vertices, {len(solid_faces)} faces")

    print(f"\nNote pad properties:")
    print(f"  Normal vector: ({notepad_normal[0]:.6f}, {notepad_normal[1]:.6f}, {notepad_normal[2]:.6f})")
    print(f"  Surface centroid: ({pan_surface_centroid[0]:.2f}, {pan_surface_centroid[1]:.2f}, {pan_surface_centroid[2]:.2f}) mm")
    print(f"  Interior centroid: ({notepad_centroid[0]:.2f}, {notepad_centroid[1]:.2f}, {notepad_centroid[2]:.2f}) mm")

    # Calculate bounding box
    bbox_min = solid_verts.min(axis=0)
    bbox_max = solid_verts.max(axis=0)
    bbox_size = bbox_max - bbox_min
    print(f"  Bounding box: {bbox_size[0]:.1f} x {bbox_size[1]:.1f} x {bbox_size[2]:.1f} mm")

    # Output paths
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    safe_note = note_name.replace('#', 's')
    obj_name = f"NotePad_{note_index}_{safe_note}{octave}"

    out_obj_path = output_dir / f"notepad_{note_index}.obj"
    stl_path = output_dir / f"notepad_{note_index}.stl"

    # Load pan centroid offset and leveling rotation for pan_body coordinate system
    import json as _json
    offset_path = Path("data/pan_centroid_offset.json")
    if offset_path.exists():
        with open(offset_path) as _f:
            _offset_data = _json.load(_f)
        pan_centroid_offset = np.array(_offset_data["centroid_offset_mm"])
    else:
        pan_centroid_offset = np.zeros(3)

    R_level = compute_leveling_rotation(obj_path)

    # Write files
    print(f"\nWriting output files...")
    # OBJ in pan_body coordinate system: center then rotate to level
    body_verts = (R_level @ (solid_verts - pan_centroid_offset).T).T
    write_obj(out_obj_path, body_verts, solid_faces, obj_name)
    # STL in mm for 3D printing
    write_stl(stl_path, solid_verts, solid_faces, obj_name)

    print(f"\n{'='*60}")
    print(f"Generated printable note pad: {note_index} ({note_name}{octave})")
    print(f"  OBJ: {out_obj_path}  (pan_body coords — overlay on pan_body.obj)")
    print(f"  STL: {stl_path}  (mm — for 3D printing)")
    print(f"  Size: {bbox_size[0]:.1f} x {bbox_size[1]:.1f} x {bbox_size[2]:.1f} mm")
    print(f"  Pan: {pan_thickness}mm thick (no groove — sits on drum surface)")
    print(f"{'='*60}")

    return {
        'index': note_index,
        'note': note_name,
        'octave': octave,
        'ring': ring,
        'normal': notepad_normal.tolist(),
        'centroid': notepad_centroid.tolist(),
        'pad_long_axis': pad_long_axis.tolist(),
        'mount_centroid': pan_surface_centroid.tolist(),
        'scale_factor': scale_factor,
        'was_scaled': was_scaled,
        'hole_positions': [pos.tolist() for pos in hole_positions],
        'hole_diameter': SCREW_HOLE_DIAMETER,
        'hole_through': True,
        'vertices': solid_verts,
        'faces': solid_faces,
        'bbox_size': bbox_size.tolist(),
        'obj_path': str(out_obj_path),
        'stl_path': str(stl_path)
    }


def save_notepad_properties(results, output_path):
    """Save note pad properties to JSON file."""
    import json

    # Extract properties (exclude large vertex/face data)
    properties = []
    for r in results:
        props = {
            'index': r['index'],
            'note': r['note'],
            'octave': r['octave'],
            'ring': r['ring'],
            'normal': r['normal'],
            'centroid': r['centroid'],
            'scale_factor': r['scale_factor'],
            'was_scaled': r['was_scaled'],
            'hole_positions': r['hole_positions'],
            'hole_diameter': r['hole_diameter'],
            'hole_through': r['hole_through'],
            'bbox_size': r['bbox_size'],
            'obj_path': r['obj_path'],
            'stl_path': r['stl_path']
        }
        properties.append(props)

    with open(output_path, 'w') as f:
        json.dump(properties, f, indent=2)

    # Report scaled pads
    scaled = [p for p in properties if p['was_scaled']]
    if scaled:
        print(f"\nScaled pads ({len(scaled)}):")
        for p in scaled:
            print(f"  {p['index']}: scaled {p['scale_factor']:.3f}x")

    print(f"Saved properties to: {output_path}")


def main():
    import sys
    obj_path = ensure_source_obj()
    output_dir = "data/notepads"

    # Parse --spread=N option (groove spread for central ring pads)
    groove_spread = 1.0
    remaining_args = []
    for arg in sys.argv[1:]:
        if arg.startswith('--spread='):
            groove_spread = float(arg.split('=')[1])
        else:
            remaining_args.append(arg)

    if groove_spread != 1.0:
        print(f"Groove spread factor: {groove_spread:.1f}x (inner ring pads)")

    # Check for command line args
    if remaining_args:
        if remaining_args[0] == '--all':
            # Generate all 29 note pads
            print("Generating all 29 note pads...")
            results = []
            # Clone map: all outer pads clone from O1 (rotate by 30° increments),
            # I4 clones from I1 (broken source geometry)
            clone_map = {
                'I4': 'I1',
                'O0': 'O1', 'O2': 'O1', 'O3': 'O1', 'O4': 'O1', 'O5': 'O1',
                'O6': 'O1', 'O7': 'O1', 'O8': 'O1', 'O9': 'O1', 'O10': 'O1', 'O11': 'O1',
            }

            for note_index in sorted(NOTE_BY_INDEX.keys()):
                clone_src = clone_map.get(note_index)
                result = generate_notepad(note_index, obj_path, output_dir,
                                         groove_spread=groove_spread,
                                         clone_from=clone_src)
                if result:
                    results.append(result)

            if results:
                # Save all properties to JSON
                save_notepad_properties(results, Path(output_dir) / "notepad_properties.json")

                # Write combined OBJ with all pads (pan_body coordinates for overlay)
                import json as _json
                offset_path = Path("data/pan_centroid_offset.json")
                if offset_path.exists():
                    with open(offset_path) as _f:
                        pan_centroid_offset = np.array(_json.load(_f)["centroid_offset_mm"])
                else:
                    pan_centroid_offset = np.zeros(3)

                R_level = compute_leveling_rotation(obj_path)

                combined_path = Path(output_dir) / "all_notepads.obj"
                with open(combined_path, 'w') as f:
                    f.write("# All note pads combined\n")
                    f.write("# Coordinates match pan_body.obj for overlay\n\n")
                    vert_offset = 0
                    for r in results:
                        idx = r['index']
                        body_verts = (R_level @ (r['vertices'] - pan_centroid_offset).T).T
                        f.write(f"o NotePad_{idx}\n")
                        for v in body_verts:
                            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                        for face in r['faces']:
                            face_str = " ".join(str(vi + 1 + vert_offset) for vi in face)
                            f.write(f"f {face_str}\n")
                        vert_offset += len(body_verts)
                        f.write("\n")
                # Generate assembly view so we can validate holes against real hardware
                print(f"\nGenerating assembly view for hole validation...")
                import generate_assembly_view
                generate_assembly_view.main()

                # Post-validate: remove any screw holes inside symmetric hardware masks
                # Uses the actual MountBase + Sleeve geometry from assembly_view
                print(f"\nValidating screw holes against hardware masks...")
                from generate_maps import compute_symmetric_hw_mask, load_assembly_view
                a_verts_v, a_objects_v = load_assembly_view()

                props_list = _json.load(open(Path(output_dir) / "notepad_properties.json"))
                total_removed = 0
                for p in props_list:
                    idx = p['index']
                    holes = p.get('hole_positions', [])
                    if not holes:
                        continue

                    # Compute mask from assembly_view hardware
                    hw_xz = []
                    for prefix in ['MountBase_', 'Sleeve_']:
                        key = prefix + idx
                        if key in a_objects_v:
                            hv = a_verts_v[sorted(a_objects_v[key])]
                            hw_xz.append(hv[:, [0, 2]])
                    pad_key = f'Pad_{idx}'
                    if not hw_xz or pad_key not in a_objects_v:
                        continue

                    pv_b = a_verts_v[sorted(a_objects_v[pad_key])]
                    mask = compute_symmetric_hw_mask(np.vstack(hw_xz), pv_b[:, [0, 2]])
                    if mask is None:
                        continue

                    # Check each hole in body coords
                    from shapely.geometry import Point as _ValPt
                    good_holes = []
                    for hp in holes:
                        h_body = R_level @ (np.array(hp) - pan_centroid_offset)
                        if not mask.contains(_ValPt(h_body[0], h_body[2])):
                            good_holes.append(hp)
                        else:
                            total_removed += 1
                    if len(good_holes) < len(holes):
                        removed = len(holes) - len(good_holes)
                        print(f"  {idx}: removed {removed} holes inside hardware mask "
                              f"({len(good_holes)} remain)")
                        # Keep all holes for pins; validated subset for drum body pilot holes
                        p['all_hole_positions'] = holes
                        p['hole_positions'] = good_holes

                if total_removed > 0:
                    print(f"  Total: removed {total_removed} holes across all pads")
                    # Re-save properties
                    with open(Path(output_dir) / "notepad_properties.json", 'w') as pf:
                        _json.dump(props_list, pf, indent=2)
                    print(f"  Re-saved notepad_properties.json")

                    # Rebuild pad OBJs for affected pads (re-subtract remaining holes)
                    print(f"  Rebuilding affected pad OBJs...")
                    for r in results:
                        idx = r['index']
                        # Find updated hole count
                        for p in props_list:
                            if p['index'] == idx:
                                new_holes = p.get('hole_positions', [])
                                if len(new_holes) < len(r.get('hole_positions', [])):
                                    # This pad had holes removed — but the OBJ
                                    # already has holes subtracted. The removed holes
                                    # just won't have pilot holes in the drum body.
                                    # The pad OBJ itself is fine (extra holes don't hurt).
                                    pass
                                break
                else:
                    print(f"  All holes validated — none in hardware zones")

                # Regenerate assembly view with updated properties
                if total_removed > 0:
                    print(f"  Regenerating assembly view...")
                    generate_assembly_view.main()

                # Generate universal screw cap (one cap fits all holes)
                print(f"\nGenerating screw hole cap...")
                cap_obj, cap_stl = generate_screw_cap(output_dir)

                print(f"\n{'='*60}")
                print(f"Generated {len(results)} note pads")
                print(f"Combined OBJ: {combined_path}  (overlay on pan_body.obj)")
                print(f"Screw cap: {cap_obj}  (press-fit, {CAP_NEEDLE_HOLE}mm needle hole)")
                print(f"Properties: {output_dir}/notepad_properties.json")
                if total_removed > 0:
                    print(f"  WARNING: {total_removed} holes removed during validation")
                print(f"{'='*60}")
            return
        else:
            # Generate specific note
            note_index = remaining_args[0]
            result = generate_notepad(note_index, obj_path, output_dir,
                                     groove_spread=groove_spread)
            if result:
                save_notepad_properties([result], Path(output_dir) / "notepad_properties.json")
            return

    # Default: generate test note O0
    result = generate_notepad("O0", obj_path, output_dir, groove_spread=groove_spread)

    if result:
        print(f"\nTest note pad generated successfully!")
        print(f"\nUsage:")
        print(f"  python generate_notepad.py                    # Generate O0 (test)")
        print(f"  python generate_notepad.py C5                 # Generate specific note")
        print(f"  python generate_notepad.py --all              # Generate all 29 notes")
        print(f"  python generate_notepad.py --all --spread=1.5 # Spread central grooves 1.5x")
        save_notepad_properties([result], Path(output_dir) / "notepad_properties.json")


if __name__ == "__main__":
    main()
