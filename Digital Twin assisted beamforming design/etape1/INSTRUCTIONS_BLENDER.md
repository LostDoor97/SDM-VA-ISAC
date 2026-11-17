# Blender tasks — step 1 (the "real" scene for DT beamforming)

Goal: build the **real world** the twin will be compared against. You add actors that carry a phone (UEs), draw their trajectories (10 s, 100 frames), and add street furniture that exists in reality but that the twin will never know about.

You do **not** re-export the whole scene. The base scene (buildings, ground, roads, the car, pedestrians A and B) stays exactly as it is in `phase3/new.xml`. Your export only contains the new objects and the trajectories. I merge them into a copy of `new.xml`.

Reference map: `etape1/carte_visibilite_BS.png`. It shows how many BS are in line of sight at 1.3 m above the ground. X/Y on the map = X/Y in Blender, in metres (I checked: the car is at (−34, 93) in both). The per-BS version is `etape1/carte_visibilite_par_BS.png`.

---

## T0. Set up the file (2 min)
1. Open **Blender 3.6** (not 5.1; 3.6 has the mitsuba add-on, though we don't use it here).
2. Open `C:\Users\AdminFix\Documents\new.blend`.
3. **File > Save As** → `C:\Users\AdminFix\Desktop\PFA\Digital Twin assisted beamforming design\scene_reelle\DT_scene_reelle.blend` (create the `scene_reelle` folder).
4. **Do not move, rename or re-material** the existing car (collection `Car`), `man_in_suit` (= pedestrian A) or `man_in_suit.001` (= pedestrian B), or any building/road. Their positions must match `new.xml`.
5. In the Outliner: right-click on Scene Collection > **New Collection**, name it exactly **`DT_nouveaux`**. Every new object goes in it; select the object and press `M` > `DT_nouveaux`.

## T1. New actors (each will carry a UE)
Naming is how my code recognises things, so **object names must be exact**.

| Object name | What | How |
|---|---|---|
| `ped_c` | Pedestrian who **stands still** (waiting at a crossing or bus stop) | Reuse `man_in_suit.002`: it already exists in the .blend but was never in `new.xml`. Move it to `DT_nouveaux` and rename it `ped_c` |
| `ped_d` | Pedestrian who walks **around a building corner** | Select `man_in_suit` > `Shift+D` > place > rename |
| `ped_e` | Pedestrian who **crosses the street** in front of a car | same |
| `car2` | Second car | Select **all** objects of the `Car` collection > `Shift+D` > move > **`Ctrl+J`** (join into one object) > rename `car2` > `M` to `DT_nouveaux` |
| `bus` *(recommended)* | Large vehicle, the classic mmWave blocker | `Add > Mesh > Cube`, scale to **12 × 2.5 × 3.2 m** (`N` panel > Dimensions), bottom at z = 0.3, material **`itu_metal`** |

Rules:
- Place each new actor **on the ground** (feet/wheels at the same height as the existing ones, z ≈ 0).
- **Orientation:** turn the actor (`R` `Z`) so it faces the direction it will travel at the start of its trajectory. My code then turns it along the curve.
- **Materials:** keep what the duplicates already have (`itu_wood`, `itu_metal`, `itu_glass`). Any new material must be named `itu_<type>`, with type one of `metal, glass, concrete, wood, brick, marble, plasterboard, ...`. The export script refuses anything else.
- The bus is deliberate: YOLO was never trained on buses, so the twin will probably **miss it**. That is a realistic twin–reality gap, and it will show up clearly in the results.

## T2. Trajectories (the important part)
One curve per moving actor, named **`traj_<actor>`**: `traj_car`, `traj_car2`, `traj_bus`, `traj_ped_a`, `traj_ped_b`, `traj_ped_d`, `traj_ped_e`. `ped_c` gets none (it stays still). You can also leave `ped_a` or `ped_b` without a curve if you want a second stationary UE.

How to draw one:
1. Select the actor > `Shift+S` > **Cursor to Selected**. The curve starts where the 3D cursor is.
2. Top view: `Numpad 7` (orthographic).
3. `Add > Curve > Bezier`. `Tab` (Edit mode) > `A` > `X` > Vertices to delete the default points. Then **`Ctrl + right-click`** to add points one after the other along the street, **starting at the actor, in the direction of travel**. Tip: Path (poly) curves or a mesh made of a chain of vertices also work.
4. `Tab` back to Object mode and rename the object `traj_car`, etc. The height of the points doesn't matter (only X/Y is used).
5. *(optional)* Speed: Object Properties > Custom Properties > `+ New` > name **`vitesse`**, value in m/s. Defaults: car 8, bus 6, pedestrians 1.4.

**Minimum length** (100 frames × 0.1 s = 10 s): car ≥ 85 m, car2 / bus ≥ 65 m, pedestrians ≥ 15 m. If a curve is shorter, the actor stops at the end. That's allowed, but say so if it's on purpose.

**What the trajectories must create.** This is the point of the step: the beamforming results only mean something if the sequence contains these events. Use the map:
- **E1 – BS switch:** at least one UE goes from a green zone (3 BS in LOS, e.g. the open square around the actors) into an orange/red zone (0–1 BS) by **turning a corner**. `ped_d` and one of the cars are the natural candidates.
- **E2 – Vehicle blockage:** the **bus** (and/or a car) drives past **right next to a UE** standing at the kerb (`ped_c`), between it and the BS that serves it best. The BS are on roofs (18–23 m), so a vehicle only cuts the line of sight if it passes within **≈ 1–3 m** of the UE. Put `ped_c` on the kerb of the lane the bus uses.
- **E3 – Body blockage:** `ped_e` walks very close (≤ 1 m) past `ped_a` or `ped_b`, or crosses right in front of a car.
- **E4 – Crossing:** `car` and `car2` cross the same intersection a few seconds apart.
- Keep everything inside the mapped area, roughly x ∈ [−170, 200], y ∈ [−180, 165], and **mostly within 150 m of at least one BS**. No actor should walk through a building or another actor.

## T3. Street furniture that exists only in reality (`clutter_*`)
The twin is built from OSM + what it perceives, so it won't know about these objects. They make the twin–reality gap realistic.
- **Lampposts:** `Add > Mesh > Cylinder`, radius 0.12 m, height 8 m, base at z = 0, material `itu_metal`. Put about **15–20** along both sides of the streets the actors use, every ~25 m. Duplicate with `Shift+D`, then select them all and `Ctrl+J` into one object named **`clutter_lampadaires`**.
- **Bus shelter** near `ped_c`: a glass box of about 4 × 1.5 × 2.5 m (`itu_glass`) plus a thin metal roof (`itu_metal`). One object named **`clutter_abribus`**.
- **2–4 traffic-light poles** at the intersection(s) you use (cylinder r = 0.1, h = 5 m, `itu_metal`), named **`clutter_feux`**.
- Everything goes in `DT_nouveaux`. Don't put clutter **on** a trajectory.

## T4. Export
1. Save (`Ctrl+S`).
2. **Scripting** workspace (top tab) > **Open** > `Digital Twin assisted beamforming design\code\blender_export_dt.py` > **Run Script** (▶).
3. Open the console (Window > Toggle System Console) and read the summary. It lists each object, each trajectory with its length, and **PROBLEMS** if there are any (wrong names, non-ITU material, curve not starting at its actor, …). Fix those and re-run until it says *"Aucun probleme detecte"*.
4. It writes `scene_reelle/meshes_dt/*.ply` and `scene_reelle/trajectoires_blender.json`. Tell me when done. A top-view screenshot (`Numpad 7`, zoomed on the area) also helps.
