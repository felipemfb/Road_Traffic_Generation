# 🚦 NYC Traffic Prediction - Graph Neural Network

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: In Development](https://img.shields.io/badge/Status-In%20Development-orange.svg)]()

Prédiction de trafic urbain par **Graph Neural Network (GNN)** avec capacité de **Transfer Learning** vers d'autres villes.

## 📋 Table des Matières

- [Objectif du Projet](#-objectif-du-projet)
- [Architecture](#-architecture)
- [Données](#-données)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Phases du Projet](#-phases-du-projet)
- [Structure des Fichiers](#-structure-des-fichiers)
- [Features Générées](#-features-générées)
- [Résultats](#-résultats)
- [Roadmap](#-roadmap)
- [Références](#-références)

---

## 🎯 Objectif du Projet

Développer un modèle GNN capable de prédire la **congestion routière** sur un segment en fonction de :
- Sa **topologie** (position dans le réseau, connexions)
- Son **contexte** (type de route, centralité, capacité)
- Le **temps** (heure, jour, saison)

### Hypothèse Centrale

> La congestion est une fonction de la **structure urbaine**, pas uniquement de l'historique local.

### Approche : Zero-Shot Transfer Learning

Le modèle est entraîné sur **New York (2010-2013)** avec des features **normalisées et relatives**, permettant son application directe à d'autres villes (ex: **Marseille**) sans ré-entraînement.

---

## 🏗️ Architecture

```
╔════════════════════════════════════════════════════════════════════════════════╗
║                              PIPELINE GLOBAL                                    ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                 ║
║    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐ ║
║    │   Phase 1    │───▶│   Phase 2    │───▶│   Phase 3    │───▶│   Phase 4   │ ║
║    │   Feature    │    │    Line      │    │     GNN      │    │ Validation  │ ║
║    │ Engineering  │    │    Graph     │    │    Model     │    │ & Transfer  │ ║
║    │     [✓]      │    │     [✓]      │    │     [ ]      │    │    [ ]      │ ║
║    └──────────────┘    └──────────────┘    └──────────────┘    └─────────────┘ ║
║                                                                                 ║
║    nodes.csv ─────────▶ nodes_enriched.csv ─────────▶ edge_index.npy           ║
║    links.csv ─────────▶ links_enriched.csv ─────────▶ node_features.npy        ║
║    travel_times.csv ──▶ travel_enriched.csv ────────▶ mean_travel_times.npy    ║
║                                                                                 ║
╚════════════════════════════════════════════════════════════════════════════════╝
```

---

## 📊 Données

### Source

Données de trafic NYC (2010-2013) basées sur les trajets de taxis :
- **Source** : [NYC Taxi Data](https://uofi.box.com/NYCtaxidata)
- **Algorithme** : [PNAS Paper](http://www.pnas.org/content/111/37/13290.full)
- **Code original** : [taxisim](https://github.com/Lab-Work/taxisim)

### Fichiers d'Entrée

| Fichier | Taille | Description |
|---------|--------|-------------|
| `nodes.csv` | ~10 MB | Intersections du réseau (coordonnées, degrés) |
| `links.csv` | ~50 MB | Segments routiers (longueur, type OSM, angles) |
| `travel_times_2013.csv` | **~5 GB** | Temps de trajet horaires (100M+ enregistrements) |

### Format des Données

```
nodes.csv:
    node_id, xcoord, ycoord, num_in_links, num_out_links, is_complete (t/f)

links.csv:
    link_id, begin_node_id, end_node_id, street_length, osm_class, begin_angle, end_angle

travel_times_2013.csv:
    begin_node_id, end_node_id, datetime, travel_time, num_trips
```

---

## 💻 Installation

### Prérequis

- Python 3.8+
- **8 GB RAM minimum** (optimisé pour traitement streaming)
- ~30 GB espace disque (données + enrichies)

### Dépendances

```bash
# Dépendances de base (Phases 1-2)
pip install pandas numpy scipy

# Dépendances GNN (Phases 3-4)
pip install torch torch-geometric networkx

# Optionnel : visualisation
pip install matplotlib seaborn
```

### Installation Complète

```bash
# Cloner le repository
git clone https://github.com/felipemfb/Road_Traffic_Generation.git
cd Road_Traffic_Generation

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

---

## 🚀 Utilisation

### Phase 1 : Feature Engineering

```bash
python enrich_graph.py \
    --data-dir ./travel_times_2013 \
    --output-dir ./enriched_data
```

**Sortie :**
- `nodes_enriched.csv` : Nœuds avec features de centralité
- `links_enriched.csv` : Liens avec features topologiques
- `travel_times_enriched.csv` : Données temporelles enrichies (~22 GB)
- `enrichment_stats.json` : Statistiques de normalisation

### Phase 2 : Line Graph Construction

```bash
python build_line_graph.py \
    --data-dir ./enriched_data \
    --output-dir ./line_graph_data

# Option : sans extraction des labels (plus rapide)
python build_line_graph.py \
    --data-dir ./enriched_data \
    --output-dir ./line_graph_data \
    --skip-labels
```

**Sortie :**
- `edge_index.npy` : Matrice d'adjacence du line graph (format COO)
- `node_features.npy` : Features des nœuds (14 dimensions)
- `node_mapping.csv` : Correspondance link_id ↔ line_graph_node_id
- `mean_travel_times.npy` : Labels (temps de trajet moyens)
- `observation_counts.npy` : Nombre d'observations par lien
- `line_graph_metadata.json` : Métadonnées et validation

### Phase 3-4 : À venir

---

## 📁 Structure des Fichiers

```
nyc-traffic-prediction/
│
├── data/
│   ├── raw/                      # Données brutes
│   │   ├── nodes.csv
│   │   ├── links.csv
│   │   └── travel_times_2013.csv
│   │
│   ├── enriched_data/            # Sortie Phase 1
│   │   ├── nodes_enriched.csv
│   │   ├── links_enriched.csv
│   │   ├── travel_times_enriched.csv
│   │   └── enrichment_stats.json
│   │
│   └── line_graph_data/          # Sortie Phase 2
│       ├── edge_index.npy
│       ├── node_features.npy
│       ├── node_mapping.csv
│       ├── mean_travel_times.npy
│       ├── observation_counts.npy
│       └── line_graph_metadata.json
│
├── scripts/
│   ├── enrich_graph.py           # Phase 1
│   ├── build_line_graph.py       # Phase 2
│   ├── train_gnn.py              # Phase 3 (à venir)
│   └── evaluate.py               # Phase 4 (à venir)
│
├── docs/
│   ├── Phase_1.pdf               # Rapport technique Phase 1
│   └── Phase_2.pdf               # Rapport technique Phase 2
│
├── README.md
└── requirements.txt
```

---

## 📈 Features Générées

### Features des Nœuds (Intersections) - Phase 1

| Feature | Type | Description |
|---------|------|-------------|
| `centrality_percentile` | [0, 1] | Position relative au centre (0=centre, 1=périphérie) |
| `degree_in_normalized` | Z-score | Nombre de liens entrants normalisé |
| `degree_out_normalized` | Z-score | Nombre de liens sortants normalisé |
| `degree_total_percentile` | [0, 1] | Degré total en rang percentile |
| `has_traffic_signal` | Binaire | Présence d'un feu de signalisation |

### Features des Liens (Segments Routiers) - Phase 1

| Feature | Type | Description |
|---------|------|-------------|
| `road_hierarchy` | {1..6} | Classification hiérarchique universelle |
| `length_percentile` | [0, 1] | Longueur relative |
| `length_zscore` | Z-score | Longueur normalisée |
| `begin_angle_sin/cos` | [-1, 1] | Orientation encodée |
| `start_centrality` | [0, 1] | Centralité du nœud de départ |
| `end_centrality` | [0, 1] | Centralité du nœud d'arrivée |
| `avg_centrality` | [0, 1] | Centralité moyenne du segment |
| `centrality_gradient` | [-1, 1] | Direction centre/périphérie |
| `connects_center` | Binaire | Segment proche du centre |
| `estimated_capacity` | [0, 1] | Capacité relative estimée |

### Features Temporelles - Phase 1

| Feature | Type | Description |
|---------|------|-------------|
| `hour_sin/cos` | [-1, 1] | Heure encodée (T=24) |
| `day_sin/cos` | [-1, 1] | Jour de la semaine (T=7) |
| `month_sin/cos` | [-1, 1] | Mois de l'année (T=12) |
| `is_weekend` | Binaire | Samedi ou dimanche |
| `is_rush_hour` | Binaire | 7-9h ou 17-19h en semaine |
| `time_period` | {1..4} | Matin/Journée/Soir/Nuit |

### Vecteur de Features du Line Graph - Phase 2

Chaque nœud du line graph (= segment routier) possède un vecteur de **14 features** :

| Index | Feature | Type |
|-------|---------|------|
| 0 | road_hierarchy | {1..6} |
| 1 | length_percentile | [0, 1] |
| 2 | length_zscore | ℝ |
| 3 | estimated_capacity | [0, 1] |
| 4 | begin_angle_sin | [-1, 1] |
| 5 | begin_angle_cos | [-1, 1] |
| 6 | angle_change | [0, 180] |
| 7 | start_centrality | [0, 1] |
| 8 | end_centrality | [0, 1] |
| 9 | avg_centrality | [0, 1] |
| 10 | centrality_gradient | [-1, 1] |
| 11 | connects_center | {0, 1} |
| 12 | start_degree_percentile | [0, 1] |
| 13 | end_degree_percentile | [0, 1] |

### Classification Hiérarchique OSM

| Niveau | Classes OSM | Description |
|--------|-------------|-------------|
| 1 | motorway, trunk | Autoroutes, voies rapides |
| 2 | primary | Routes principales |
| 3 | secondary | Routes secondaires |
| 4 | tertiary | Routes tertiaires |
| 5 | residential, living_street | Voies résidentielles |
| 6 | service, unclassified | Voies de service |

---

## 📊 Résultats

### Phase 1 - Feature Engineering

| Métrique | Valeur |
|----------|--------|
| Nœuds traités | 96,435 |
| Liens traités | 260,855 |
| Enregistrements temporels | ~100,000,000 |
| Taux de correspondance | 98.5% |
| Temps d'exécution | ~15 min |

### Phase 2 - Line Graph Construction

| Métrique | Valeur |
|----------|--------|
| Nœuds du line graph | 260,855 |
| Arêtes du line graph | 805,415 |
| Degré moyen | 3.09 |
| Nœuds isolés | 4 |
| Features par nœud | 14 |
| Liens avec données trafic | 58,267 (22.3%) |
| Observations moyennes/lien | 1,781 |
| Temps d'exécution | ~1h15 |
| Mémoire edge_index | 6.4 MB |
| Mémoire node_features | 14.6 MB |

### Chargement PyTorch Geometric

```python
import torch
import numpy as np
from torch_geometric.data import Data

# Charger les données
edge_index = torch.from_numpy(np.load('edge_index.npy')).long()
x = torch.from_numpy(np.load('node_features.npy')).float()
y = torch.from_numpy(np.load('mean_travel_times.npy')).float()

# Créer l'objet Data
data = Data(x=x, edge_index=edge_index, y=y)

print(f"Nœuds: {data.num_nodes}")        # 260,855
print(f"Arêtes: {data.num_edges}")       # 805,415
print(f"Features: {data.num_node_features}")  # 14
```

---

## 🗺️ Roadmap

- [x] **v0.1** - Phase 1 : Feature Engineering
- [x] **v0.2** - Phase 2 : Line Graph Construction
- [ ] **v0.3** - Phase 3 : GNN Model (GraphSAGE/GAT)
- [ ] **v0.4** - Phase 4 : Validation NYC (split spatial)
- [ ] **v1.0** - Transfer Learning vers Marseille

---

## 📚 Références

### Papers

1. Donovan, B., & Work, D. B. (2015). *Using coarse GPS data to quantify city-scale transportation system resilience to extreme events*. PNAS.

2. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs*. NeurIPS.

3. Veličković, P., et al. (2018). *Graph Attention Networks*. ICLR.

### Datasets

- [NYC Taxi Data](https://uofi.box.com/NYCtaxidata)
- [OpenStreetMap](https://www.openstreetmap.org/)

### Outils

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [NetworkX](https://networkx.org/)
- [taxisim](https://github.com/Lab-Work/taxisim)

---

## 📄 License

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 👥 Contributeurs

- **Caramella Enzo** - *Développeur principal*

---

## 📧 Contact


Pour les données originales : bpdonov2@illinois.edu

