"""
Phase 3 (new scene) export fixup.

Sionna only auto-recognizes a BSDF as a radio material if its id (after
stripping the Blender exporter's "mat-" prefix) starts with "itu_" AND the
remainder EXACTLY matches one of its 14 known ITU-R P.2040 material keys
(concrete, brick, plasterboard, wood, glass, ceiling_board, chipboard,
plywood, marble, floorboard, metal, very_dry_ground, medium_dry_ground,
wet_ground) -- see sionna/rt/scene_utils.py. A custom name like "car_wood"
fails both conditions and would make load_scene() error out with
"which is not a radio material".

We deliberately gave the car / pedestrian A / pedestrian B their own
DISTINCT material names in Blender (car_wood, ped_a_wood, ped_B_wood) to stop
the exporter from welding all three into one shared mesh (cf. the merge bug
we hit earlier -- shapes sharing one material name get combined into a
single mesh, which breaks independent per-actor movement). This script
reconciles the two requirements: it renames each to start with "itu_"
(satisfies Sionna's naming gate) while injecting an explicit
`<string name="type" value="...">` override so each still resolves to the
correct underlying ITU material's EM properties instead of failing a lookup
on its own made-up name.

Run this after every Blender export, before loading the scene in Sionna --
same role as phase3/fix_scene2_export.py played for the previous scene.

Usage:
    .venv\\Scripts\\python.exe phase3\\fix_new_export.py [path/to/new.xml]
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# custom Blender material name -> ITU preset it should actually use
CUSTOM_MATERIAL_MAP = {
    "car_wood": "wood",
    "ped_a_wood": "wood",
    "ped_B_wood": "wood",
}


def fix_export(path: Path):
    tree = ET.parse(path)
    root = tree.getroot()

    n_fixed = 0
    for custom_name, itu_type in CUSTOM_MATERIAL_MAP.items():
        old_id = f"mat-{custom_name}"
        new_id = f"mat-itu_{custom_name}"

        # Find the bsdf definition (searches both top-level and nested-in-shape,
        # matching the same search Sionna itself does).
        bsdf = None
        for candidate in root.findall("./bsdf") + root.findall(".//shape/bsdf"):
            if candidate.attrib.get("id") == old_id:
                bsdf = candidate
                break
        if bsdf is None:
            print(f"  (skip) no bsdf found with id={old_id!r} -- already fixed, "
                  f"or this material isn't in the export.")
            continue

        # Rename the bsdf definition itself.
        bsdf.attrib["id"] = new_id
        if bsdf.attrib.get("name") == old_id:
            bsdf.attrib["name"] = new_id

        # Inject the type override as a DIRECT CHILD of this bsdf element
        # (Sionna's parser only looks at direct children for this).
        type_el = ET.Element("string", {"name": "type", "value": itu_type})
        bsdf.insert(0, type_el)

        # Update every <ref id="mat-..."> that pointed at the old id.
        n_refs = 0
        for ref in root.findall(".//ref"):
            if ref.attrib.get("id") == old_id:
                ref.attrib["id"] = new_id
                n_refs += 1

        print(f"  patched {old_id} -> {new_id} (itu_type={itu_type}), "
              f"{n_refs} reference(s) updated")
        n_fixed += 1

    tree.write(path, encoding="utf-8", xml_declaration=True)
    print(f"\n{n_fixed} material(s) patched -> {path}")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("phase3/new.xml")
    assert target.exists(), f"File not found: {target}"
    fix_export(target)
