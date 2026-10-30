#!/usr/bin/env python
"""Prepare ManiSkill's ``panda_stick`` URDF for Isaac Sim's URDF importer.

The importer silently drops the ``.glb`` visual meshes of the Franka links, so this script exports every glb
part as an OBJ (one file per material part, node transforms applied) and writes a URDF copy whose visuals
reference the OBJs with the glb base colours as URDF ``<material>`` colours. Collision STLs are referenced
in place. The result is cached under ``burger/backends/isaac/assets/generated/panda_stick_isaac/``.

  .venv-isaac61/bin/python burger/backends/isaac/tools/prepare_robot_assets.py [--force]
"""
from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"
SRC_URDF = ASSET_DIR / "maniskill" / "panda_stick.urdf"
OUT_DIR = ASSET_DIR / "generated" / "panda_stick_isaac"
OUT_URDF = OUT_DIR / "panda_stick_isaac.urdf"


def glb_to_obj_parts(glb_path: Path, out_dir: Path, stem: str) -> list[tuple[Path, tuple[float, float, float, float]]]:
    """Export each geometry node of a glb as an OBJ with its node transform applied; returns (path, rgba) pairs."""
    import numpy as np
    import trimesh

    scene = trimesh.load(str(glb_path))
    if not isinstance(scene, trimesh.Scene):
        scene = trimesh.Scene(scene)
    parts = []
    for i, node in enumerate(scene.graph.nodes_geometry):
        T, gname = scene.graph[node]
        mesh = scene.geometry[gname].copy()
        mesh.apply_transform(T)
        mat = getattr(mesh.visual, "material", None)
        color = getattr(mat, "baseColorFactor", None)
        if color is None:
            color = getattr(mat, "main_color", None)
        rgba = tuple(float(c) / 255.0 for c in (np.asarray(color)[:4] if color is not None else (204, 204, 204, 255)))
        if len(rgba) == 3:
            rgba = (*rgba, 1.0)
        out = out_dir / f"{stem}_{i}.obj"
        # plain geometry only: colours come from the URDF material, which the importer honours
        out.write_text(trimesh.exchange.obj.export_obj(mesh, include_normals=True, include_color=False, include_texture=False, write_texture=False, header=None))
        parts.append((out, rgba))
    return parts


def prepare(force: bool = False, log=print) -> Path:
    if OUT_URDF.exists() and not force:
        return OUT_URDF
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mesh_dir = OUT_DIR / "meshes"
    mesh_dir.mkdir(exist_ok=True)
    tree = ET.parse(SRC_URDF)
    root = tree.getroot()
    n_visual = 0
    for link in root.findall("link"):
        visuals = link.findall("visual")
        for vis in visuals:
            mesh = vis.find("geometry/mesh")
            if mesh is None or not mesh.get("filename", "").endswith(".glb"):
                continue
            glb = (SRC_URDF.parent / mesh.get("filename")).resolve()
            parts = glb_to_obj_parts(glb, mesh_dir, glb.stem)
            origin = vis.find("origin")
            link.remove(vis)
            for j, (obj, rgba) in enumerate(parts):
                v = ET.SubElement(link, "visual", {"name": f"{glb.stem}_{j}"})
                if origin is not None:
                    v.append(origin)
                g = ET.SubElement(v, "geometry")
                ET.SubElement(g, "mesh", {"filename": str(obj.relative_to(OUT_DIR))})
                m = ET.SubElement(v, "material", {"name": f"{glb.stem}_{j}_mat"})
                ET.SubElement(m, "color", {"rgba": " ".join(f"{c:.4f}" for c in rgba)})
                n_visual += 1
        for col in link.findall("collision"):
            mesh = col.find("geometry/mesh")
            if mesh is not None and mesh.get("filename", "").endswith(".stl"):
                src = (SRC_URDF.parent / mesh.get("filename")).resolve()
                dst = mesh_dir / src.name
                if not dst.exists():
                    shutil.copy(src, dst)
                mesh.set("filename", str(dst.relative_to(OUT_DIR)))
    # the stick: give the URDF cylinder visual an explicit colour too (ManiSkill renders it with the default grey material)
    for link in root.findall("link"):
        for vis in link.findall("visual"):
            if vis.find("geometry/cylinder") is not None and vis.find("material") is None:
                m = ET.SubElement(vis, "material", {"name": "stick_mat"})
                ET.SubElement(m, "color", {"rgba": "0.8 0.8 0.8 1.0"})
    tree.write(OUT_URDF, encoding="utf-8", xml_declaration=True)
    log(f"[prepare_robot_assets] wrote {OUT_URDF} with {n_visual} OBJ visuals")
    return OUT_URDF


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    prepare(force=a.force)
