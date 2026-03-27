#!/usr/bin/env python3
"""
Structural cylinder and radial strut parameters.

The drum interior is supported by:
- A single cylinder (R=115..135mm), split at R=125mm for assembly
- 6 radial struts at 60deg intervals (aligned with section boundaries)

Assembly: 7 pieces total
  1 central piece (R < 125mm): inner cylinder wall + strut hubs, 6 vertical slots
  6 outer pieces (R >= 125mm): outer cylinder arc + strut segments, with keys

Joinery (all full-height):
  - Radial key/slot at R=125mm, at each strut angle: 10mm wide, 2.5mm deep
  - Side key/slot on strut edges: 2.5mm deep, full radial length
"""

import math

# ── Cylinder (split at CYL_SPLIT_R for assembly) ─────────────
CYL_OUTER_R = 135.0            # mm
CYL_INNER_R = 115.0            # mm
CYL_SPLIT_R = 125.0            # mm — assembly cut radius

# ── Radial struts ─────────────────────────────────────────────
N_STRUTS = 6
STRUT_WALL = 20.0              # mm tangential thickness
STRUT_HALF_W = STRUT_WALL / 2  # 10mm
# Strut angles: section boundaries (gaps between outer pads)
STRUT_ANGLES_DEG = [15.0, 75.0, 135.0, 195.0, 255.0, 315.0]
STRUT_ANGLES_RAD = [math.radians(a) for a in STRUT_ANGLES_DEG]

# ── Key/slot (full height) ───────────────────────────────────
KEY_WIDTH = 10.0               # mm — tangential span of radial keys
KEY_DEPTH = 2.5                # mm — protrusion depth

# ── Wiring holes (30mm diameter, horizontal) ──────────────────
WIRE_HOLE_R = 15.0             # mm radius (30mm dia)
WIRE_HOLE_CENTER_HEIGHT = 30.0 # mm above base plate
# Strut wiring holes: one per strut at this radial position
STRUT_WIRE_R_POS = 175.0      # mm
# Cylinder wiring holes: 6, at sector midpoints (between struts)
WIRE_CYL_ANGLES_DEG = [45.0, 105.0, 165.0, 225.0, 285.0, 345.0]
WIRE_CYL_ANGLES_RAD = [math.radians(a) for a in WIRE_CYL_ANGLES_DEG]

# ── Screw holes (M3, base plate attachment) ───────────────────
SCREW_PILOT_R = 1.25           # M3 tapped pilot radius
SCREW_CLEAR_R = 1.7            # M3 clearance radius
SCREW_DEPTH = 8.0              # pilot hole depth into material
SCREW_EDGE_MAX = 5.0           # max distance from nearest edge
# Cylinder screws: two rings near edges, 6 per ring at sector midpoints
CYL_SCREW_R_INNER = CYL_INNER_R + SCREW_EDGE_MAX   # 120mm
CYL_SCREW_R_OUTER = CYL_OUTER_R - SCREW_EDGE_MAX   # 130mm
CYL_SCREW_N = 6               # per ring
# Strut screws: two rows per strut (5mm from each tangential edge)
STRUT_SCREW_SPACING = 40.0    # mm between screws along strut length
STRUT_SCREW_R_START = CYL_OUTER_R + 10  # 145mm — start past cylinder
