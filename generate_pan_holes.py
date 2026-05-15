"""
Extract the pan body with pad and groove faces removed, leaving a
"pan_holes" mesh: the drum wall + inter-pad surface, with holes where
each note pad + its surrounding groove used to be.

Output: data/quarters/pan_holes.obj
"""
import numpy as np
from pathlib import Path

from generate_sector import extract_bowl_surface
from generate_quarter import compute_pan_rotation, write_quarter_obj
from generate_notepad import NOTE_MAPPING

OBJ_PATH = "data/Tenor Pan only.obj"
OUTPUT_PATH = Path("data/quarters/pan_holes.obj")


def main():
    print("Loading bowl surface...")
    bowl_v, bowl_f, face_mat, face_grp, _ = extract_bowl_surface(OBJ_PATH)

    print("Computing orientation rotation (drum wall vertical along +Z)...")
    R = compute_pan_rotation(bowl_v, bowl_f, face_mat)
    bowl_v = bowl_v @ R.T

    pad_groups = {pan_g for (_, pan_g) in NOTE_MAPPING.keys()}
    # Groove faces stay in the body so the drum surface extends into the
    # groove space — the separately-generated groove solids overlap rather
    # than butt against the body. Avoids hairline gaps after voxel remesh.
    remove_groups = pad_groups
    print(f"  Removing {len(pad_groups)} pad groups (groove groups kept in body)")

    kept_faces = [f for f, g in zip(bowl_f, face_grp) if g not in remove_groups]
    print(f"  Kept {len(kept_faces):,} / {len(bowl_f):,} faces")

    used = {vi for face in kept_faces for vi in face}
    old_to_new = {old: new for new, old in enumerate(sorted(used))}
    new_verts = np.array([bowl_v[old] for old in sorted(used)])
    new_faces = [[old_to_new[vi] for vi in face] for face in kept_faces]
    print(f"  {len(new_verts):,} verts, {len(new_faces):,} faces")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_quarter_obj(OUTPUT_PATH, new_verts, new_faces, name="PanHoles")


if __name__ == "__main__":
    main()
