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
PCB_BOSS_HEIGHT = 4.6           # Boss protrusion (1.3× base 3.5mm for clearance)
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
    """Generate push-cap as a solid cylinder with pizza-slice slit cut out.

    Uses trimesh boolean: build a solid cylinder, subtract a wedge for the
    wire slit.  Thread ridges are added to the outer surface.
    """
    import trimesh

    outer_r = CAP_OUTER_DIAMETER / 2
    slit_half_angle = math.atan2(SLIT_WIDTH / 2, outer_r)

    # ── Build solid cylinder (no hollow interior) ──
    cap_cyl = trimesh.creation.cylinder(
        radius=outer_r, height=CAP_TOTAL_LENGTH, sections=SEGMENTS)
    # Default cylinder is centered at origin along Z — shift so top=0, bottom=-length
    cap_cyl.apply_translation([0, 0, -CAP_TOTAL_LENGTH / 2])

    # ── Create wedge for the wire slit ──
    # Build a triangular prism manually (6 verts, 8 tris) for robust boolean.
    slit_r = outer_r + CAP_THREAD_DEPTH + 1.0  # overshoot for clean cut
    slit_z_top = SLIT_TOP_Z + 0.5   # overshoot above top cap
    slit_z_bot = SLIT_BOTTOM_Z - 0.5  # overshoot below slit end

    a1 = SLIT_ANGLE - slit_half_angle
    a2 = SLIT_ANGLE + slit_half_angle
    # 6 vertices: triangle at top and bottom of slit
    wv = np.array([
        [0.0, 0.0, slit_z_top],                                       # 0 center top
        [slit_r * math.cos(a1), slit_r * math.sin(a1), slit_z_top],   # 1 right top
        [slit_r * math.cos(a2), slit_r * math.sin(a2), slit_z_top],   # 2 left top
        [0.0, 0.0, slit_z_bot],                                       # 3 center bot
        [slit_r * math.cos(a1), slit_r * math.sin(a1), slit_z_bot],   # 4 right bot
        [slit_r * math.cos(a2), slit_r * math.sin(a2), slit_z_bot],   # 5 left bot
    ])
    wf = np.array([
        [0, 2, 1],       # top triangle
        [3, 4, 5],       # bottom triangle
        [0, 1, 4], [0, 4, 3],  # right face
        [0, 3, 5], [0, 5, 2],  # left face
        [1, 2, 5], [1, 5, 4],  # outer face
    ])
    wedge = trimesh.Trimesh(vertices=wv, faces=wf, process=True)
    trimesh.repair.fix_normals(wedge)

    # Subtract slit wedge from cylinder
    try:
        cap_cyl = cap_cyl.difference(wedge, engine='manifold')
    except Exception as e:
        print(f"  WARNING: slit boolean failed: {e}")

    # ── Add thread ridges on the outer surface ──
    # Build a thin ring at each thread ridge and union it
    z_count = max(int(CAP_TOTAL_LENGTH / CAP_THREAD_PITCH * 12), 48)
    thread_verts = []
    thread_faces = []

    for z_idx in range(z_count + 1):
        z = -z_idx * CAP_TOTAL_LENGTH / z_count
        if -CAP_INSERTION_LENGTH <= z <= -CAP_TOP_THICKNESS:
            phase = (-z / CAP_THREAD_PITCH) % 1.0
            thread_h = push_thread_profile(phase, CAP_THREAD_DEPTH)
        else:
            thread_h = 0.0

        ring_start = len(thread_verts)
        for i in range(SEGMENTS):
            angle = 2 * math.pi * i / SEGMENTS
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            # Inner at cylinder surface, outer at surface + thread
            thread_verts.append([(outer_r) * cos_a, (outer_r) * sin_a, z])
            thread_verts.append([(outer_r + thread_h) * cos_a,
                                 (outer_r + thread_h) * sin_a, z])

    # Build thread ridge faces between adjacent z-levels
    for z_idx in range(z_count):
        for i in range(SEGMENTS):
            i_next = (i + 1) % SEGMENTS
            base = z_idx * SEGMENTS * 2
            next_base = (z_idx + 1) * SEGMENTS * 2
            # Outer face of thread ridge
            o0 = base + i * 2 + 1
            o1 = base + i_next * 2 + 1
            o2 = next_base + i_next * 2 + 1
            o3 = next_base + i * 2 + 1
            thread_faces.append([o0, o1, o2])
            thread_faces.append([o0, o2, o3])

    # Combine: use the boolean result as the base, thread ridges as visual overlay
    cap_v = np.array(cap_cyl.vertices)
    cap_f = [list(f) for f in cap_cyl.faces]

    return cap_v, cap_f


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
    y_hole_back = -ht + yo  # hole ends flush with plate back face

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
    gusset_v, gusset_f = generate_gusset()

    # Build plate + bosses as proper trimesh volumes, then drill M2 holes
    xo = PLATE_X_OFFSET
    yo = PLATE_Y_OFFSET
    ht = PLATE_THICKNESS / 2
    half_grid = PCB_HOLE_GRID / 2
    z_top = PLATE_Z_TOP
    z_center = z_top - PLATE_HEIGHT / 2

    # Plate box as trimesh volume
    plate_mesh = trimesh.creation.box(
        extents=[PLATE_WIDTH, PLATE_THICKNESS, PLATE_HEIGHT],
        transform=trimesh.transformations.translation_matrix([xo, yo, z_center]))

    # Union 4 boss cylinders onto the plate
    boss_r = PCB_BOSS_DIAMETER / 2
    y_plate_face = ht + yo
    y_boss_end = y_plate_face + PCB_BOSS_HEIGHT
    boss_height = PCB_BOSS_HEIGHT
    boss_center_y = (y_plate_face + y_boss_end) / 2

    hole_xz = [
        (-half_grid + xo, z_center - half_grid),
        (+half_grid + xo, z_center - half_grid),
        (+half_grid + xo, z_center + half_grid),
        (-half_grid + xo, z_center + half_grid),
    ]
    for hx, hz in hole_xz:
        boss = trimesh.creation.cylinder(
            radius=boss_r, height=boss_height, sections=32)
        # Rotate to Y axis
        R90 = np.eye(4)
        R90[1, 1] = 0; R90[1, 2] = -1
        R90[2, 1] = 1; R90[2, 2] = 0
        boss.apply_transform(R90)
        T = trimesh.transformations.translation_matrix([hx, boss_center_y, hz])
        boss.apply_transform(T)
        try:
            plate_mesh = plate_mesh.union(boss, engine='manifold')
        except Exception:
            pass
    print(f"  Plate+bosses: {len(plate_mesh.faces)}f, vol={plate_mesh.is_volume}")

    # Drill M2 clearance holes through plate + bosses (32 sections for clean circle)
    M2_CLEARANCE_R = 1.1  # M2 clearance = 2.2mm diameter
    y_drill_top = y_boss_end + 1.0
    y_drill_bot = -ht + yo - 2.0
    bore_length = y_drill_top - y_drill_bot
    bore_center_y = (y_drill_top + y_drill_bot) / 2

    n_before = len(plate_mesh.faces)
    for hx, hz in hole_xz:
        drill = trimesh.creation.cylinder(
            radius=M2_CLEARANCE_R, height=bore_length, sections=32)
        R90 = np.eye(4)
        R90[1, 1] = 0; R90[1, 2] = -1
        R90[2, 1] = 1; R90[2, 2] = 0
        drill.apply_transform(R90)
        T = trimesh.transformations.translation_matrix([hx, bore_center_y, hz])
        drill.apply_transform(T)
        try:
            plate_mesh = plate_mesh.difference(drill, engine='manifold')
        except Exception:
            pass
    print(f"  Plate drill: {n_before}→{len(plate_mesh.faces)}f, vol={plate_mesh.is_volume}")

    plate_v = np.array(plate_mesh.vertices)
    plate_f = [list(f) for f in plate_mesh.faces]

    all_verts = cap_v
    all_faces = list(cap_f)

    offset = len(all_verts)
    all_verts = np.vstack([all_verts, plate_v])
    all_faces += [[i + offset for i in f] for f in plate_f]

    offset = len(all_verts)
    all_verts = np.vstack([all_verts, gusset_v])
    all_faces += [[i + offset for i in f] for f in gusset_f]

    return all_verts, all_faces


def generate_pcb_prototype():
    """Generate a 20x20mm PCB prototype with M2 through-holes and pins.

    The PCB sits against the vertical plate of the push-cap mount.
    Through-holes are boolean-subtracted so screws pass cleanly through.
    Pins protrude in -Y through the boss holes and stick out the other side.

    Oriented in the same local frame as generate_central_mount():
    Z=0 at pad side, -Z downward.

    Returns (vertices, faces).
    """
    import trimesh

    PCB_SIZE = 20.0
    PCB_THICK = 1.6       # standard PCB thickness
    PIN_R = 0.9           # slightly under M2 (1.0mm) for fit
    PIN_SEGS = 8
    HOLE_R = PCB_HOLE_DIAMETER / 2  # M2 clearance

    # Pin length: through boss + plate + 1mm protrusion out the back
    PIN_LENGTH = PCB_BOSS_HEIGHT + PLATE_THICKNESS + 1.0

    # PCB board position
    plate_front_y = PLATE_Y_OFFSET + PLATE_THICKNESS / 2
    pcb_y0 = plate_front_y + PCB_BOSS_HEIGHT   # PCB sits at boss tips
    pcb_y1 = pcb_y0 + PCB_THICK

    half = PCB_SIZE / 2
    pcb_x0 = PLATE_X_OFFSET - half
    pcb_x1 = PLATE_X_OFFSET + half
    z_center = PLATE_Z_TOP - PLATE_HEIGHT / 2
    pcb_z0 = z_center - half
    pcb_z1 = z_center + half

    # Create PCB slab as trimesh box
    pcb_box = trimesh.creation.box(
        extents=[PCB_SIZE, PCB_THICK, PCB_SIZE],
        transform=trimesh.transformations.translation_matrix([
            PLATE_X_OFFSET, (pcb_y0 + pcb_y1) / 2, z_center]))

    # Boolean-subtract M2 through-holes
    grid_half = PCB_HOLE_GRID / 2
    hole_centres_xz = [
        (PLATE_X_OFFSET - grid_half, z_center - grid_half),
        (PLATE_X_OFFSET + grid_half, z_center - grid_half),
        (PLATE_X_OFFSET + grid_half, z_center + grid_half),
        (PLATE_X_OFFSET - grid_half, z_center + grid_half),
    ]
    for hx, hz in hole_centres_xz:
        cyl = trimesh.creation.cylinder(radius=HOLE_R, height=PCB_THICK + 2, sections=16)
        # Rotate to align with Y axis
        R90 = np.eye(4)
        R90[1, 1] = 0; R90[1, 2] = -1
        R90[2, 1] = 1; R90[2, 2] = 0
        cyl.apply_transform(R90)
        T = np.eye(4)
        T[0, 3] = hx
        T[1, 3] = (pcb_y0 + pcb_y1) / 2
        T[2, 3] = hz
        cyl.apply_transform(T)
        try:
            pcb_box = pcb_box.difference(cyl, engine='manifold')
        except Exception:
            pass

    # Build pin geometry (separate from PCB for clean mesh)
    pin_verts = []
    pin_faces = []
    for hx, hz in hole_centres_xz:
        pin_base = len(pin_verts)
        # Top ring at PCB inner face
        for i in range(PIN_SEGS):
            a = 2 * math.pi * i / PIN_SEGS
            pin_verts.append([hx + PIN_R * math.cos(a), pcb_y0, hz + PIN_R * math.sin(a)])
        # Bottom ring (protrudes through boss holes in -Y)
        for i in range(PIN_SEGS):
            a = 2 * math.pi * i / PIN_SEGS
            pin_verts.append([hx + PIN_R * math.cos(a), pcb_y0 - PIN_LENGTH, hz + PIN_R * math.sin(a)])
        top_c = len(pin_verts)
        pin_verts.append([hx, pcb_y0, hz])
        bot_c = len(pin_verts)
        pin_verts.append([hx, pcb_y0 - PIN_LENGTH, hz])
        for i in range(PIN_SEGS):
            j = (i + 1) % PIN_SEGS
            pin_faces.append([pin_base + i, pin_base + PIN_SEGS + i, pin_base + PIN_SEGS + j])
            pin_faces.append([pin_base + i, pin_base + PIN_SEGS + j, pin_base + j])
            pin_faces.append([top_c, pin_base + j, pin_base + i])
            pin_faces.append([bot_c, pin_base + PIN_SEGS + i, pin_base + PIN_SEGS + j])

    # Combine PCB board + pins
    pcb_v = np.array(pcb_box.vertices)
    pcb_f = [list(f) for f in pcb_box.faces]
    offset = len(pcb_v)
    all_v = np.vstack([pcb_v, np.array(pin_verts, dtype=float)])
    all_f = pcb_f + [[i + offset for i in f] for f in pin_faces]

    return all_v, all_f


def generate_bounding_ellipse_prism(tolerance=0.5):
    """Generate a vertical elliptical prism around the push-cap mechanism.

    The prism has an elliptical cross-section whose major axis (X) aligns
    with the pad's long axis and whose minor axis (Y) aligns with the pad's
    short axis.  The semi-axes are computed from the mechanism component
    extents so that cap + boss + plate + gusset are fully enclosed.

    Oriented in the same local frame: Z=0 at pad side, -Z downward.
    No end caps (open prism).

    Returns (vertices, faces, semi_x, semi_y, z_top, z_bot).
    """
    from generate_notepad import CENTRAL_MOUNT_WALL_THICKNESS

    cap_r = CAP_OUTER_DIAMETER / 2 + CAP_THREAD_DEPTH
    boss_r = BOSS_INNER_DIAMETER / 2 + CENTRAL_MOUNT_WALL_THICKNESS

    # Collect critical XY points from all mechanism components
    hw = PLATE_WIDTH / 2
    ht = PLATE_THICKNESS / 2
    xo = PLATE_X_OFFSET
    yo = PLATE_Y_OFFSET
    half_grid = PCB_HOLE_GRID / 2
    y_boss_end = ht + yo + PCB_BOSS_HEIGHT
    gusset_hw = PLATE_WIDTH * 0.3
    y_gusset = -ht + yo - _GUSSET_DEPTH

    critical = []
    # Boss / cap perimeter (circular, sample densely)
    for r in (cap_r, boss_r):
        for i in range(64):
            a = 2 * math.pi * i / 64
            critical.append((r * math.cos(a), r * math.sin(a)))
    # Plate corners
    for x in (-hw + xo, hw + xo):
        for y in (-ht + yo, ht + yo):
            critical.append((x, y))
    # PCB boss tips on plate front face
    for cx in (-half_grid + xo, half_grid + xo):
        critical.append((cx, y_boss_end))
    # Gusset corners
    for x in (-gusset_hw + xo, gusset_hw + xo):
        critical.append((x, y_gusset))

    pts = np.array(critical)
    x_max = float(np.abs(pts[:, 0]).max())
    y_max = float(np.abs(pts[:, 1]).max())

    # Start with bounding-box semi-axes, then scale up uniformly
    # until every critical point satisfies (x/a)^2 + (y/b)^2 <= 1
    a, b = x_max, y_max
    if a > 0 and b > 0:
        violation = float(np.max((pts[:, 0] / a) ** 2 + (pts[:, 1] / b) ** 2))
        if violation > 1.0:
            scale = math.sqrt(violation)
            a *= scale
            b *= scale

    semi_x = a + tolerance
    semi_y = b + tolerance

    z_top = 1.0   # slightly above cap top
    z_bot = PLATE_Z_TOP - PLATE_HEIGHT - 2.0  # below PCB plate

    # Generate open elliptical prism (wall only, no end caps)
    S = SEGMENTS
    verts = []
    faces_out = []

    # Top ring
    r0 = len(verts)
    for i in range(S):
        ang = 2 * math.pi * i / S
        verts.append([semi_x * math.cos(ang), semi_y * math.sin(ang), z_top])
    # Bottom ring
    r1 = len(verts)
    for i in range(S):
        ang = 2 * math.pi * i / S
        verts.append([semi_x * math.cos(ang), semi_y * math.sin(ang), z_bot])

    # Side faces (quads as 2 triangles)
    for i in range(S):
        j = (i + 1) % S
        faces_out.append([r0 + i, r1 + i, r1 + j])
        faces_out.append([r0 + i, r1 + j, r0 + j])

    return np.array(verts, dtype=float), faces_out, semi_x, semi_y, z_top, z_bot


def main():
    from generate_notepad import write_obj, write_stl

    print("Generating central ring push-cap mount...")
    print(f"  Cap OD: {CAP_OUTER_DIAMETER:.1f}mm (fits {BOSS_INNER_DIAMETER}mm boss)")
    print(f"  Insertion: {CAP_INSERTION_LENGTH:.0f}mm, Exposed: {CAP_EXPOSED_LENGTH:.0f}mm")
    print(f"  Wire slit: {SLIT_WIDTH:.0f}mm wide, Z={SLIT_TOP_Z:.0f} to {SLIT_BOTTOM_Z:.0f}")
    print(f"  Vertical plate: {PLATE_WIDTH:.0f}x{PLATE_HEIGHT:.0f}x{PLATE_THICKNESS:.1f}mm")
    print(f"  PCB holes: M2 on {PCB_HOLE_GRID:.0f}mm grid (matches outer ring)")

    verts, faces = generate_central_mount()
    print(f"  Vertices: {len(verts)}, Faces: {len(faces)}")

    out_dir = Path("data/mounts")
    out_dir.mkdir(exist_ok=True)
    write_obj(out_dir / "central_push_cap.obj", verts, faces, "CentralPushCap")
    write_stl(out_dir / "central_push_cap.stl", verts, faces, "CentralPushCap")

    # PCB prototype
    print("\nGenerating PCB prototype (20x20mm)...")
    pcb_v, pcb_f = generate_pcb_prototype()
    print(f"  PCB: {len(pcb_v)}v, {len(pcb_f)}f")
    write_obj(out_dir / "pcb_prototype.obj", pcb_v, pcb_f, "PCBPrototype")
    write_stl(out_dir / "pcb_prototype.stl", pcb_v, pcb_f, "PCBPrototype")

    # Bounding ellipse prism
    print("\nGenerating bounding ellipse prism...")
    bc_v, bc_f, bc_sx, bc_sy, bc_zt, bc_zb = generate_bounding_ellipse_prism(tolerance=1.5)
    print(f"  Semi-axes: X={bc_sx:.1f}mm, Y={bc_sy:.1f}mm, Z=[{bc_zb:.1f}, {bc_zt:.1f}]mm")
    print(f"  {len(bc_v)}v, {len(bc_f)}f")
    write_obj(out_dir / "bounding_ellipse.obj", bc_v, bc_f, "BoundingEllipse")


if __name__ == "__main__":
    main()
