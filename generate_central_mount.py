#!/usr/bin/env python3
"""
Generate push-cap mount for central/inner ring pads (Adafruit 1740 sensor).

Design:
  - Cylindrical cap with external push-fit thread ridges
  - Solid closed top (sits inside boss against pad)
  - Wire slit cut through the cap wall for draping the sensor wire out
  - Solid closed bottom cap connecting to the vertical PCB plate
  - Small grip tabs (not large wings) for finger purchase during mounting
  - Vertical PCB base plate with M2 holes matching outer ring layout

Oriented with Z=0 at the top (pad side), extending downward along -Z.
"""

import numpy as np
from pathlib import Path
import math

# ── Dimensions matching the pad boss ─────────────────────────────────────
BOSS_INNER_DIAMETER = 16.0        # matches generate_notepad.py CENTRAL_MOUNT_INNER_DIAMETER
BOSS_THREAD_DEPTH = 0.3           # shallow internal ridges (easy push)
BOSS_THREAD_PITCH = 2.0
BOSS_DEPTH = 10.0

# ── Push-cap body ────────────────────────────────────────────────────────
THREAD_CLEARANCE = 0.3
CAP_OUTER_DIAMETER = BOSS_INNER_DIAMETER - THREAD_CLEARANCE
CAP_WALL_THICKNESS = 1.5
CAP_INNER_DIAMETER = CAP_OUTER_DIAMETER - 2 * CAP_WALL_THICKNESS
CAP_THREAD_DEPTH = 0.3            # shallow external ridges (matches boss)
CAP_THREAD_PITCH = BOSS_THREAD_PITCH
CAP_INSERTION_LENGTH = 8.0
CAP_EXPOSED_LENGTH = 4.0
CAP_TOTAL_LENGTH = CAP_INSERTION_LENGTH + CAP_EXPOSED_LENGTH

# Solid top and bottom caps
CAP_TOP_THICKNESS = 1.5
CAP_BOTTOM_THICKNESS = 2.0

# ── Wire slit (slot through wall AND top cap for draping wire) ───────────
SLIT_WIDTH = 4.0                # Angular width of slit
SLIT_TOP_Z = 0.1                # Extends through the top cap (above z=0)
SLIT_BOTTOM_Z = -(CAP_INSERTION_LENGTH - 1.0)  # Ends 1mm before boss exit
SLIT_ANGLE = 0.0                # At angle = 0

# ── Grip tabs (small nubs for finger grip) ───────────────────────────────
GRIP_LENGTH = 4.0               # How far tabs extend outward
GRIP_WIDTH = 6.0                # Circumferential width of each tab
GRIP_THICKNESS = 2.5            # Vertical thickness
GRIP_Z = -(CAP_INSERTION_LENGTH + 1.0)  # Just below boss exit
GRIP_COUNT = 2                  # Two tabs, opposite sides (90° from slit)

# ── Vertical PCB base plate ─────────────────────────────────────────────
# Centre the plate vertically on the cap so PCB fits compactly under the pad.
# The cap top is at z=0 (pad underside), insertion ends at -CAP_INSERTION_LENGTH,
# exposed part extends to -CAP_TOTAL_LENGTH.
# We centre the plate on the mid-point of the exposed portion.
PLATE_WIDTH = 22.0              # Wide enough for 16mm hole grid + margin
PLATE_HEIGHT = 22.0             # Tall enough for 16mm hole grid + margin
PLATE_THICKNESS = 2.5
PLATE_Z_TOP = -CAP_TOTAL_LENGTH  # Plate hangs from cap bottom
PLATE_X_OFFSET = 0.0            # Centred on cap axis in X
# Shift plate+gusset in +Y so gusset tip sits at -cap_radius (cap edge)
# Gusset tip = plate_back - gusset_depth = (-ht + yo) - gusset_depth
# Want gusset tip at Y = -cap_radius
# So: -ht + yo - gusset_depth = -cap_radius
# yo = -cap_radius + ht + gusset_depth
_GUSSET_DEPTH = 4.0
PLATE_Y_OFFSET = -CAP_OUTER_DIAMETER / 2 + PLATE_THICKNESS / 2 + _GUSSET_DEPTH

# PCB mounting — matches mount_base.py hole pattern
PCB_HOLE_DIAMETER = 2.2         # M2 clearance (same as mount_base)
PCB_BOSS_DIAMETER = 5.0         # Boss OD (same as mount_base)
PCB_BOSS_HEIGHT = 3.5           # Boss protrusion reduced (plate is part of screw path)
PCB_HOLE_GRID = 16.0            # Grid spacing (same as mount_base)
BOSS_SEGMENTS = 16
PCB_HOLE_GRID = 16.0            # Grid spacing (matches outer ring)
BOSS_SEGMENTS = 16

SEGMENTS = 32


def push_thread_profile(phase, depth):
    """Sawtooth push-fit ridge: gentle ramp in, steep retention."""
    phase = phase % 1.0
    if phase < 0.6:
        return depth * (phase / 0.6)
    elif phase < 0.7:
        return depth
    else:
        return depth * (1.0 - (phase - 0.7) / 0.3)


def generate_push_cap():
    """Generate push-cap cylinder with solid top, solid bottom, wire slit, and threads."""
    vertices = []
    faces = []

    outer_r = CAP_OUTER_DIAMETER / 2
    inner_r = CAP_INNER_DIAMETER / 2

    slit_half_angle = math.atan2(SLIT_WIDTH / 2, outer_r)

    def in_slit(seg_idx, z):
        """Check if this segment/z is in the wire slit."""
        if z > SLIT_TOP_Z or z < SLIT_BOTTOM_Z:
            return False
        angle = 2 * math.pi * seg_idx / SEGMENTS
        diff = abs(angle - SLIT_ANGLE)
        if diff > math.pi:
            diff = 2 * math.pi - diff
        return diff <= slit_half_angle

    z_count = max(int(CAP_TOTAL_LENGTH / CAP_THREAD_PITCH * 12), 48)

    outer_rings = []
    inner_rings = []

    for z_idx in range(z_count + 1):
        z = -z_idx * CAP_TOTAL_LENGTH / z_count

        # Thread ridges only in insertion region, below top cap, above bottom cap
        if -CAP_INSERTION_LENGTH <= z <= -CAP_TOP_THICKNESS:
            phase = (-z / CAP_THREAD_PITCH) % 1.0
            thread_h = push_thread_profile(phase, CAP_THREAD_DEPTH)
        else:
            thread_h = 0.0

        outer_ring = {}
        inner_ring = {}

        for i in range(SEGMENTS):
            # Skip segments in slit region
            if in_slit(i, z):
                continue

            angle = 2 * math.pi * i / SEGMENTS
            cos_a, sin_a = math.cos(angle), math.sin(angle)

            r_out = outer_r + thread_h

            outer_ring[i] = len(vertices)
            vertices.append([r_out * cos_a, r_out * sin_a, z])

            inner_ring[i] = len(vertices)
            vertices.append([inner_r * cos_a, inner_r * sin_a, z])

        outer_rings.append(outer_ring)
        inner_rings.append(inner_ring)

    # Build faces for wall segments that exist at both z levels
    for z_idx in range(z_count):
        segs_both = sorted(set(outer_rings[z_idx].keys()) & set(outer_rings[z_idx + 1].keys()))
        for s_idx in range(len(segs_both) - 1):
            i, i_next = segs_both[s_idx], segs_both[s_idx + 1]
            # Only connect adjacent segments (skip gaps)
            if (i_next - i) > 2:
                continue
            # Outer wall
            faces.append([outer_rings[z_idx][i], outer_rings[z_idx + 1][i],
                         outer_rings[z_idx + 1][i_next], outer_rings[z_idx][i_next]])
            # Inner wall
            faces.append([inner_rings[z_idx][i], inner_rings[z_idx][i_next],
                         inner_rings[z_idx + 1][i_next], inner_rings[z_idx + 1][i]])

    # ── Solid top cap (z=0) — full disk ──
    top_center = len(vertices)
    vertices.append([0, 0, 0])
    top_segs = sorted(outer_rings[0].keys())
    for s_idx in range(len(top_segs) - 1):
        i, i_next = top_segs[s_idx], top_segs[s_idx + 1]
        if (i_next - i) > 2:
            continue
        faces.append([top_center, outer_rings[0][i], outer_rings[0][i_next]])
    # Close the last-to-first if they're adjacent
    if len(top_segs) > 2:
        first, last = top_segs[0], top_segs[-1]
        if (SEGMENTS - last + first) <= 2:
            faces.append([top_center, outer_rings[0][last], outer_rings[0][first]])

    # ── Solid bottom cap (z = -CAP_TOTAL_LENGTH) — full disk ──
    bot_center = len(vertices)
    vertices.append([0, 0, -CAP_TOTAL_LENGTH])
    bot_segs = sorted(outer_rings[-1].keys())
    for s_idx in range(len(bot_segs) - 1):
        i, i_next = bot_segs[s_idx], bot_segs[s_idx + 1]
        if (i_next - i) > 2:
            continue
        faces.append([bot_center, outer_rings[-1][i_next], outer_rings[-1][i]])
    if len(bot_segs) > 2:
        first, last = bot_segs[0], bot_segs[-1]
        if (SEGMENTS - last + first) <= 2:
            faces.append([bot_center, outer_rings[-1][first], outer_rings[-1][last]])

    # ── Slit edge walls (seal the sides of the slit opening) ──
    for z_idx in range(z_count):
        segs_this = sorted(outer_rings[z_idx].keys())
        segs_next = sorted(outer_rings[z_idx + 1].keys())
        # Find edges where segments appear/disappear (slit boundaries)
        for segs, ring_z, ring_z_label in [(segs_this, z_idx, 'this'), (segs_next, z_idx + 1, 'next')]:
            pass  # Slit walls are handled implicitly by missing segments

    return np.array(vertices), faces


def generate_grip_tabs():
    """Generate small grip tabs for finger purchase."""
    vertices = []
    faces = []

    cap_r = CAP_OUTER_DIAMETER / 2
    grip_r = cap_r + GRIP_LENGTH

    z_top = GRIP_Z
    z_bot = GRIP_Z - GRIP_THICKNESS

    for wing_angle_deg in [90, 270]:
        wing_angle = math.radians(wing_angle_deg)
        half_w = math.atan2(GRIP_WIDTH / 2, cap_r)

        a1 = wing_angle - half_w
        a2 = wing_angle + half_w

        b = len(vertices)

        # Top face: inner-left, inner-right, outer-right, outer-left
        vertices.append([cap_r * math.cos(a1), cap_r * math.sin(a1), z_top])
        vertices.append([cap_r * math.cos(a2), cap_r * math.sin(a2), z_top])
        vertices.append([grip_r * math.cos(a2), grip_r * math.sin(a2), z_top])
        vertices.append([grip_r * math.cos(a1), grip_r * math.sin(a1), z_top])
        # Bottom face
        vertices.append([cap_r * math.cos(a1), cap_r * math.sin(a1), z_bot])
        vertices.append([cap_r * math.cos(a2), cap_r * math.sin(a2), z_bot])
        vertices.append([grip_r * math.cos(a2), grip_r * math.sin(a2), z_bot])
        vertices.append([grip_r * math.cos(a1), grip_r * math.sin(a1), z_bot])

        faces.append([b+0, b+1, b+2, b+3])  # Top
        faces.append([b+4, b+7, b+6, b+5])  # Bottom
        faces.append([b+3, b+2, b+6, b+7])  # Outer
        faces.append([b+1, b+0, b+4, b+5])  # Inner
        faces.append([b+0, b+3, b+7, b+4])  # Left
        faces.append([b+2, b+1, b+5, b+6])  # Right

    return np.array(vertices), faces


def generate_vertical_plate():
    """Generate vertical PCB base plate with hollow M2 screw bosses on 16mm grid.
    Bosses protrude from the +Y face of the plate, identical to the outer ring
    mount base bosses (5mm OD, 2.2mm ID, 6mm tall)."""
    vertices = []
    faces = []

    hw = PLATE_WIDTH / 2
    ht = PLATE_THICKNESS / 2
    xo = PLATE_X_OFFSET
    yo = PLATE_Y_OFFSET          # Shift toward gusset side so gusset tip = cap edge
    z_top = PLATE_Z_TOP
    z_bot = PLATE_Z_TOP - PLATE_HEIGHT

    # Plate box (shifted by xo in X, yo in Y)
    b = len(vertices)
    for z in [z_top, z_bot]:
        for x, y in [(-hw + xo, -ht + yo), (hw + xo, -ht + yo), (hw + xo, ht + yo), (-hw + xo, ht + yo)]:
            vertices.append([x, y, z])

    faces.append([b+3, b+2, b+6, b+7])  # Front (+Y)
    faces.append([b+0, b+4, b+5, b+1])  # Back (-Y)
    faces.append([b+0, b+1, b+2, b+3])  # Top
    faces.append([b+4, b+7, b+6, b+5])  # Bottom
    faces.append([b+0, b+3, b+7, b+4])  # Left (-X)
    faces.append([b+2, b+1, b+5, b+6])  # Right (+X)

    # ── Hollow screw bosses on +Y face (matches mount_base.py exactly) ──
    half_grid = PCB_HOLE_GRID / 2
    boss_r = PCB_BOSS_DIAMETER / 2
    hole_r = PCB_HOLE_DIAMETER / 2
    z_center = z_top - PLATE_HEIGHT / 2

    y_plate_face = ht + yo
    y_boss_end = y_plate_face + PCB_BOSS_HEIGHT
    y_hole_back = -ht + yo - 0.5  # hole extends through plate back (+ overshoot)

    boss_positions = [
        (-half_grid + xo, z_center - half_grid),
        (+half_grid + xo, z_center - half_grid),
        (+half_grid + xo, z_center + half_grid),
        (-half_grid + xo, z_center + half_grid),
    ]

    def add_ring(cx, cz, radius, y):
        start = len(vertices)
        for i in range(BOSS_SEGMENTS):
            angle = 2 * math.pi * i / BOSS_SEGMENTS
            vertices.append([cx + radius * math.cos(angle), y,
                           cz + radius * math.sin(angle)])
        return start

    def connect_rings(r1, r2, reverse=False):
        for i in range(BOSS_SEGMENTS):
            i0, i1 = r1 + i, r1 + (i + 1) % BOSS_SEGMENTS
            j0, j1 = r2 + i, r2 + (i + 1) % BOSS_SEGMENTS
            if reverse:
                faces.append([i0, i1, j1, j0])
            else:
                faces.append([i0, j0, j1, i1])

    def connect_annulus(outer_start, inner_start, flip=False):
        for i in range(BOSS_SEGMENTS):
            o0 = outer_start + i
            o1 = outer_start + (i + 1) % BOSS_SEGMENTS
            i0 = inner_start + i
            i1 = inner_start + (i + 1) % BOSS_SEGMENTS
            if flip:
                faces.append([o0, o1, i1, i0])
            else:
                faces.append([o0, i0, i1, o1])

    for cx, cz in boss_positions:
        # Boss outer wall (plate face to boss tip)
        outer_base = add_ring(cx, cz, boss_r, y_plate_face)
        outer_tip = add_ring(cx, cz, boss_r, y_boss_end)
        connect_rings(outer_base, outer_tip)

        # M2 hole — continuous bore from boss tip right through plate back
        inner_tip = add_ring(cx, cz, hole_r, y_boss_end)
        inner_face = add_ring(cx, cz, hole_r, y_plate_face)
        inner_back = add_ring(cx, cz, hole_r, y_hole_back)
        connect_rings(inner_tip, inner_face, reverse=True)   # tip to plate face
        connect_rings(inner_face, inner_back, reverse=True)   # plate face to back

        # Boss annulus at plate face (outer boss to hole bore)
        connect_annulus(outer_base, inner_face, flip=True)

        # Boss annulus at tip (outer boss to hole bore)
        connect_annulus(outer_tip, inner_tip)

        # No back cap — hole exits clean through the plate back

    return np.array(vertices), faces


def generate_gusset():
    """Generate a triangular wedge gusset on the back (-Y) side of the
    vertical plate, connecting the cap exit edge to the plate for strength.

    The gusset is a right triangle:
      - Top edge: at cap exit (z = -CAP_INSERTION_LENGTH)
      - Vertical edge: down the back of the plate
      - Hypotenuse: diagonal brace
    """
    vertices = []
    faces = []

    xo = PLATE_X_OFFSET
    yo = PLATE_Y_OFFSET
    ht = PLATE_THICKNESS / 2
    gusset_depth = _GUSSET_DEPTH
    gusset_height = 16.0

    z_top = -CAP_INSERTION_LENGTH  # Gusset starts at cap exit
    z_bot = z_top - gusset_height
    y_plate_back = -ht + yo        # Back face of plate (shifted)
    y_gusset = y_plate_back - gusset_depth  # Gusset tip

    gusset_width = PLATE_WIDTH * 0.6  # Narrower than plate
    hw = gusset_width / 2

    # Triangle wedge: 6 vertices (2 triangular cross-sections, left and right)
    b = len(vertices)
    # Left side (x = xo - hw)
    vertices.append([xo - hw, y_plate_back, z_top])   # 0: top at plate
    vertices.append([xo - hw, y_gusset,     z_top])   # 1: top at gusset tip
    vertices.append([xo - hw, y_plate_back, z_bot])    # 2: bottom at plate

    # Right side (x = xo + hw)
    vertices.append([xo + hw, y_plate_back, z_top])   # 3: top at plate
    vertices.append([xo + hw, y_gusset,     z_top])   # 4: top at gusset tip
    vertices.append([xo + hw, y_plate_back, z_bot])    # 5: bottom at plate

    # Left triangle face
    faces.append([b+0, b+2, b+1])
    # Right triangle face
    faces.append([b+3, b+4, b+5])
    # Top face (horizontal)
    faces.append([b+0, b+1, b+4, b+3])
    # Back face (plate side, vertical)
    faces.append([b+0, b+3, b+5, b+2])
    # Hypotenuse face (diagonal)
    faces.append([b+1, b+2, b+5, b+4])

    return np.array(vertices), faces


def generate_central_mount():
    """Generate the complete push-cap assembly.
    Returns (vertices, faces) for the combined mesh."""
    import trimesh

    cap_v, cap_f = generate_push_cap()
    grip_v, grip_f = generate_grip_tabs()
    plate_v, plate_f = generate_vertical_plate()
    gusset_v, gusset_f = generate_gusset()

    # Boolean-subtract M2 holes from the plate box so holes go right through
    xo = PLATE_X_OFFSET
    yo = PLATE_Y_OFFSET
    ht = PLATE_THICKNESS / 2
    half_grid = PCB_HOLE_GRID / 2
    z_top = PLATE_Z_TOP
    z_center = z_top - PLATE_HEIGHT / 2
    hole_r = PCB_HOLE_DIAMETER / 2

    # Build plate as trimesh
    tri_pf = []
    for f in plate_f:
        if len(f) == 3: tri_pf.append(f)
        elif len(f) == 4: tri_pf.append([f[0],f[1],f[2]]); tri_pf.append([f[0],f[2],f[3]])
    plate_mesh = trimesh.Trimesh(vertices=plate_v, faces=np.array(tri_pf), process=True)
    trimesh.repair.fix_normals(plate_mesh)

    hole_xz = [
        (-half_grid + xo, z_center - half_grid),
        (+half_grid + xo, z_center - half_grid),
        (+half_grid + xo, z_center + half_grid),
        (-half_grid + xo, z_center + half_grid),
    ]

    for hx, hz in hole_xz:
        cyl = trimesh.creation.cylinder(radius=hole_r, height=PLATE_THICKNESS + 2, sections=16)
        # Rotate cylinder to align with Y axis (plate thickness direction)
        T = np.eye(4)
        T[0, 3] = hx
        T[1, 3] = yo  # at plate center Y
        T[2, 3] = hz
        # Rotate 90° about X to align cylinder Z with world Y
        R90 = np.eye(4)
        R90[1, 1] = 0; R90[1, 2] = -1
        R90[2, 1] = 1; R90[2, 2] = 0
        cyl.apply_transform(R90)
        cyl.apply_transform(T)
        try:
            plate_mesh = plate_mesh.difference(cyl, engine='manifold')
        except Exception:
            pass

    plate_v = np.array(plate_mesh.vertices)
    plate_f = [list(f) for f in plate_mesh.faces]

    all_verts = cap_v
    all_faces = list(cap_f)

    offset = len(all_verts)
    all_verts = np.vstack([all_verts, grip_v])
    all_faces += [[i + offset for i in f] for f in grip_f]

    offset = len(all_verts)
    all_verts = np.vstack([all_verts, plate_v])
    all_faces += [[i + offset for i in f] for f in plate_f]

    offset = len(all_verts)
    all_verts = np.vstack([all_verts, gusset_v])
    all_faces += [[i + offset for i in f] for f in gusset_f]

    return all_verts, all_faces


def main():
    from generate_notepad import write_obj, write_stl

    print("Generating central ring push-cap mount...")
    print(f"  Cap OD: {CAP_OUTER_DIAMETER:.1f}mm (fits {BOSS_INNER_DIAMETER}mm boss)")
    print(f"  Insertion: {CAP_INSERTION_LENGTH:.0f}mm, Exposed: {CAP_EXPOSED_LENGTH:.0f}mm")
    print(f"  Wire slit: {SLIT_WIDTH:.0f}mm wide, Z={SLIT_TOP_Z:.0f} to {SLIT_BOTTOM_Z:.0f}")
    print(f"  Grip tabs: {GRIP_LENGTH:.0f}mm extension, {GRIP_WIDTH:.0f}mm wide")
    print(f"  Vertical plate: {PLATE_WIDTH:.0f}x{PLATE_HEIGHT:.0f}x{PLATE_THICKNESS:.1f}mm")
    print(f"  PCB holes: M2 on {PCB_HOLE_GRID:.0f}mm grid (matches outer ring)")

    verts, faces = generate_central_mount()
    print(f"  Vertices: {len(verts)}, Faces: {len(faces)}")

    out_dir = Path("data/mounts")
    out_dir.mkdir(exist_ok=True)
    write_obj(out_dir / "central_push_cap.obj", verts, faces, "CentralPushCap")
    write_stl(out_dir / "central_push_cap.stl", verts, faces, "CentralPushCap")


if __name__ == "__main__":
    main()
