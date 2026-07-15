# SDM-VA-ISAC — Phase 2 : Simulation Sionna RT

Ce dossier contient le pipeline de simulation de propagation radio (ray tracing) exécuté sur la scène 3D `riad.xml`, dans le cadre de la Phase 2 du projet SDM-VA-ISAC.

## 1. Contenu du dossier

| Fichier / dossier | Description |
|---|---|
| `sionna_rt_osm_pipeline.ipynb` | Notebook principal : charge la scène, place les antennes, lance le ray tracing, calcule les CIR/PDP et la carte de couverture. |
| `riad.xml` | Scène 3D (format Mitsuba) générée à partir des données OpenStreetMap (`osm/`). |
| `install_env.ps1` | Script d'installation de l'environnement Python (dépendances). |
| `osm/` | Données OSM sources ayant servi à générer la scène. |
| `meshes/` | Maillages 3D des bâtiments utilisés dans `riad.xml`. |
| `sionna/`, `mitsuba-blender/` | Bibliothèques installées localement en mode développement (sources de Sionna RT et de l'add-on Blender). |
| `radio_map_riad.png`, `pdp_ue0_bs0.png`, `pdp_all_ue.png` | Figures générées par le notebook (carte de couverture, profils de délai de puissance). |
| `phase2_sionna_rt_summary.json` | Résultats numériques exportés, servant d'entrée à la Phase 3. |

## 2. Prérequis

- Windows avec **PowerShell**
- **Python 3.11**
- **GPU NVIDIA avec CUDA** (le solveur de ray tracing utilise `cuda_ad_mono_polarized`)
- [`uv`](https://github.com/astral-sh/uv) installé (`pip install uv` si nécessaire)
- Une connexion internet pour le téléchargement des dépendances (~5-6 Go, dont PyTorch avec CUDA)

## 3. Installation de l'environnement

À faire **une seule fois**, depuis le dossier `PFA` :

```powershell
cd C:\Users\AdminFix\Desktop\PFA
.\install_env.ps1
```

Ce script :
1. Installe PyTorch (CUDA 12.4)
2. Installe Mitsuba 3.8.0 + DrJIT 1.3.1 (moteur de rendu/ray tracing)
3. Installe `sionna-rt` en local (module de ray tracing radio)
4. Installe `sionna` en local, sans ses dépendances par défaut
5. Installe Jupyter et enregistre un kernel dédié `PFA (Python 3.11)`

Durée estimée : 10 à 20 minutes selon la connexion.

À la fin, le script affiche les versions installées pour vérification (torch, mitsuba, sionna, sionna.rt).

## 4. Lancer la simulation

Depuis le dossier `PFA` :

```powershell
cd C:\Users\AdminFix\Desktop\PFA
.venv\Scripts\jupyter notebook sionna_rt_osm_pipeline.ipynb
```

Dans Jupyter, sélectionner le kernel **PFA (Python 3.11)**, puis exécuter les cellules dans l'ordre (`Run All` ou cellule par cellule avec `Shift+Enter`).

### Structure du notebook

1. **Imports et configuration** — chargement des bibliothèques et du variant GPU Mitsuba.
2. **Chargement de la scène `riad.xml`** — import de la géométrie 3D du riad.
3. **Configuration des antennes et placement des équipements** — définition des réseaux d'antennes et positionnement des stations de base (BS) et des utilisateurs (UE).
4. **Lancer de rayons (Ray Tracing)** — calcul des trajets de propagation entre BS et UE via `PathSolver`.
5. **Extraction de la réponse impulsionnelle (CIR)** — conversion des trajets en réponse impulsionnelle en bande de base.
6. **Visualisation du Power Delay Profile (PDP)** — tracé de la puissance reçue en fonction du délai, pour une paire BS/UE.
7. **Comparaison PDP — tous les UE** — superposition des PDP de tous les utilisateurs.
8. **Carte de couverture radio (RadioMapSolver)** — génération de la carte de puissance reçue sur toute la scène.
9. **Paramètres ISAC** — calcul des sorties nécessaires à la Phase 3 (détection/communication conjointes).
10. **Export Phase 2 → JSON** — sauvegarde des résultats dans `phase2_sionna_rt_summary.json`, en entrée de la Phase 3.

## 5. Modifier le placement des stations de base

Le placement des BS/UE se fait dans la cellule de la section **3. Configuration des réseaux d'antennes et placement des équipements**, via des coordonnées `[x, y, z]` (en mètres) dans le repère de la scène. La hauteur `z` doit être choisie au-dessus des toits des bâtiments environnants (visible via la carte de couverture ou l'aperçu de la scène `scene.preview()`).

## 6. Résultats attendus

À l'issue de l'exécution complète, le notebook produit :
- `radio_map_riad.png` : carte de couverture radio de la scène
- `pdp_ue0_bs0.png`, `pdp_all_ue.png` : profils de délai de puissance
- `phase2_sionna_rt_summary.json` : résumé numérique des résultats de la Phase 2, utilisé comme point de départ de la Phase 3
