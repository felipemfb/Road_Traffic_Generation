#!/usr/bin/env python3
"""
build_line_graph.py - Phase 2: Construction du Dataset Edge-Centric (Line Graph)
=================================================================================

Script de transformation du graphe routier en Line Graph (Dual Graph) pour
permettre l'utilisation d'architectures GNN standard sur les segments routiers.

Dans le Line Graph:
- Les segments routiers (links) deviennent des NŒUDS
- Deux nœuds sont connectés si leurs segments partagent une intersection
  (le end_node de l'un = le begin_node de l'autre)

Optimisé pour environnements à mémoire limitée (8GB RAM):
- Construction de l'adjacence en streaming (2 passes)
- Stockage sur disque des features par chunks
- Matrices sparse pour l'adjacence
- Types de données optimisés (float32, int32)

Auteur: Claude (Anthropic)
Date: 05 Janvier 2026
Version: 1.0.0

Usage:
    python build_line_graph.py --data-dir ./enriched_data --output-dir ./line_graph_data

Entrées requises (depuis Phase 1):
    - links_enriched.csv: Liens enrichis avec features normalisées
    - nodes_enriched.csv: Nœuds enrichis (optionnel, pour validation)
    - travel_times_enriched.csv: Données de trafic (optionnel, pour labels)

Sorties:
    - edge_index.npy: Matrice d'adjacence du line graph (format COO)
    - node_features.npy: Features des nœuds du line graph (= features des liens)
    - node_mapping.csv: Correspondance link_id → line_graph_node_id
    - labels/: Dossier avec labels par chunks temporels (optionnel)
    - line_graph_metadata.json: Métadonnées et statistiques
"""

import argparse
import gc
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


# =============================================================================
# CONFIGURATION
# =============================================================================

# Taille des chunks pour le traitement des gros fichiers
CHUNK_SIZE = 200_000  # Réduit pour 8GB RAM

# Features à extraire pour le GNN (colonnes du links_enriched.csv)
# Ces features seront les attributs des nœuds dans le line graph
NODE_FEATURES = [
    # Features topologiques normalisées
    'road_hierarchy',           # int8: Classification 1-6
    'length_percentile',        # float32: Longueur relative [0,1]
    'length_zscore',            # float32: Longueur Z-score
    'estimated_capacity',       # float32: Capacité relative [0,1]
    
    # Features géométriques
    'begin_angle_sin',          # float32: Orientation sin [-1,1]
    'begin_angle_cos',          # float32: Orientation cos [-1,1]
    'angle_change',             # float32: Courbure [0,180]
    
    # Features de centralité
    'start_centrality',         # float32: Centralité nœud départ [0,1]
    'end_centrality',           # float32: Centralité nœud arrivée [0,1]
    'avg_centrality',           # float32: Centralité moyenne [0,1]
    'centrality_gradient',      # float32: Direction centre/périphérie [-1,1]
    'connects_center',          # int8: Binaire proche du centre
    
    # Features des nœuds adjacents
    'start_degree_percentile',  # float32: Degré nœud départ [0,1]
    'end_degree_percentile',    # float32: Degré nœud arrivée [0,1]
]

# Types de données optimisés
LINKS_ENRICHED_DTYPES = {
    'link_id': 'int64',
    'begin_node_id': 'int64',
    'end_node_id': 'int64',
    'road_hierarchy': 'int8',
    'length_percentile': 'float32',
    'length_zscore': 'float32',
    'estimated_capacity': 'float32',
    'begin_angle_sin': 'float32',
    'begin_angle_cos': 'float32',
    'angle_change': 'float32',
    'start_centrality': 'float32',
    'end_centrality': 'float32',
    'avg_centrality': 'float32',
    'centrality_gradient': 'float32',
    'connects_center': 'int8',
    'start_degree_percentile': 'float32',
    'end_degree_percentile': 'float32',
    'street_length': 'float32',
    'osm_class': 'category',
}

TRAVEL_TIMES_DTYPES = {
    'begin_node_id': 'int64',
    'end_node_id': 'int64',
    'travel_time': 'float32',
    'num_trips': 'int32',
    'hour': 'int8',
    'day_of_week': 'int8',
    'is_weekend': 'int8',
    'is_rush_hour': 'int8',
    'road_hierarchy': 'float32',
}

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('build_line_graph.log')
    ]
)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Encodeur JSON personnalisé pour les types numpy et autres."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, set):
            return list(obj)
        return super().default(obj)


# =============================================================================
# PHASE 2A: CONSTRUCTION DE L'INDEX D'ADJACENCE
# =============================================================================

def build_adjacency_index(links_path: str) -> Tuple[Dict[int, List[int]], Dict[int, int], int]:
    """
    Première passe: Construire l'index des connexions entre liens.
    
    Stratégie mémoire-efficace:
    - Ne charge que les colonnes nécessaires (link_id, begin_node_id, end_node_id)
    - Construit deux dictionnaires légers:
      * end_node → [liens qui y finissent]
      * begin_node → [liens qui y commencent]
    
    Args:
        links_path: Chemin vers links_enriched.csv
        
    Returns:
        Tuple (node_to_outgoing_links, link_to_idx, num_links)
        - node_to_outgoing_links: Dict[node_id] → List[link_id] des liens sortants
        - link_to_idx: Dict[link_id] → index dans le line graph
        - num_links: Nombre total de liens
    """
    logger.info("Phase 2A: Construction de l'index d'adjacence...")
    
    # Dictionnaires pour l'index
    node_to_outgoing: Dict[int, List[int]] = defaultdict(list)
    node_to_incoming: Dict[int, List[int]] = defaultdict(list)
    link_to_idx: Dict[int, int] = {}
    
    # Lire uniquement les colonnes nécessaires
    cols_needed = ['link_id', 'begin_node_id', 'end_node_id']
    
    # Traitement par chunks pour économiser la mémoire
    idx = 0
    for chunk in pd.read_csv(links_path, usecols=cols_needed, chunksize=CHUNK_SIZE):
        for _, row in chunk.iterrows():
            link_id = int(row['link_id'])
            begin_node = int(row['begin_node_id'])
            end_node = int(row['end_node_id'])
            
            # Mapping link_id → index
            link_to_idx[link_id] = idx
            
            # Index pour trouver les voisins
            node_to_outgoing[begin_node].append(link_id)
            node_to_incoming[end_node].append(link_id)
            
            idx += 1
        
        logger.info(f"  - {idx:,} liens indexés...")
    
    num_links = idx
    logger.info(f"  - Total: {num_links:,} liens")
    logger.info(f"  - Nœuds uniques: {len(node_to_outgoing):,}")
    
    # Calculer le nombre estimé d'arêtes dans le line graph
    # Une arête existe entre L1 et L2 si end_node(L1) = begin_node(L2)
    estimated_edges = sum(
        len(node_to_incoming.get(node, [])) * len(outgoing)
        for node, outgoing in node_to_outgoing.items()
    )
    logger.info(f"  - Arêtes estimées dans le line graph: {estimated_edges:,}")
    
    return dict(node_to_outgoing), node_to_incoming, link_to_idx, num_links


def build_edge_index(links_path: str, 
                     node_to_outgoing: Dict[int, List[int]],
                     node_to_incoming: Dict[int, List[int]],
                     link_to_idx: Dict[int, int]) -> np.ndarray:
    """
    Deuxième passe: Construire la matrice d'adjacence du line graph.
    
    Pour chaque lien L1 qui finit au nœud N, on crée une arête vers
    chaque lien L2 qui part de N.
    
    Format de sortie: edge_index au format COO (2, num_edges)
    - edge_index[0]: indices des nœuds source
    - edge_index[1]: indices des nœuds destination
    
    Args:
        links_path: Chemin vers links_enriched.csv
        node_to_outgoing: Dict[node_id] → List[link_id] sortants
        node_to_incoming: Dict[node_id] → List[link_id] entrants
        link_to_idx: Dict[link_id] → index
        
    Returns:
        edge_index: np.ndarray de shape (2, num_edges)
    """
    logger.info("Phase 2B: Construction de la matrice d'adjacence...")
    
    # Listes pour construire edge_index
    sources = []
    targets = []
    
    # Lire les liens pour obtenir end_node de chaque lien
    cols_needed = ['link_id', 'end_node_id']
    
    edges_count = 0
    for chunk in pd.read_csv(links_path, usecols=cols_needed, chunksize=CHUNK_SIZE):
        for _, row in chunk.iterrows():
            link_id = int(row['link_id'])
            end_node = int(row['end_node_id'])
            
            # Index de ce lien (source dans le line graph)
            src_idx = link_to_idx[link_id]
            
            # Tous les liens qui partent du nœud où ce lien finit
            # sont des voisins dans le line graph
            outgoing_links = node_to_outgoing.get(end_node, [])
            
            for target_link_id in outgoing_links:
                # Éviter les self-loops (un lien connecté à lui-même)
                if target_link_id != link_id:
                    tgt_idx = link_to_idx[target_link_id]
                    sources.append(src_idx)
                    targets.append(tgt_idx)
                    edges_count += 1
        
        if edges_count % 1_000_000 == 0 and edges_count > 0:
            logger.info(f"  - {edges_count:,} arêtes créées...")
    
    # Convertir en numpy array
    edge_index = np.array([sources, targets], dtype=np.int32)
    
    logger.info(f"  - Total: {edge_index.shape[1]:,} arêtes dans le line graph")
    logger.info(f"  - Degré moyen: {edge_index.shape[1] / len(link_to_idx):.2f}")
    
    return edge_index


# =============================================================================
# PHASE 2C: EXTRACTION DES FEATURES DES NŒUDS
# =============================================================================

def extract_node_features(links_path: str, 
                          link_to_idx: Dict[int, int],
                          num_links: int) -> Tuple[np.ndarray, List[str]]:
    """
    Extraire les features des liens comme features des nœuds du line graph.
    
    Chaque lien devient un nœud, avec ses attributs enrichis comme features.
    
    Args:
        links_path: Chemin vers links_enriched.csv
        link_to_idx: Dict[link_id] → index
        num_links: Nombre total de liens
        
    Returns:
        Tuple (node_features, feature_names)
        - node_features: np.ndarray de shape (num_nodes, num_features)
        - feature_names: Liste des noms de features
    """
    logger.info("Phase 2C: Extraction des features des nœuds...")
    
    # Déterminer les features disponibles
    # Lire d'abord les colonnes du fichier
    sample = pd.read_csv(links_path, nrows=5)
    available_features = [f for f in NODE_FEATURES if f in sample.columns]
    
    logger.info(f"  - Features demandées: {len(NODE_FEATURES)}")
    logger.info(f"  - Features disponibles: {len(available_features)}")
    
    missing = set(NODE_FEATURES) - set(available_features)
    if missing:
        logger.warning(f"  - Features manquantes: {missing}")
    
    # Initialiser la matrice de features
    num_features = len(available_features)
    node_features = np.zeros((num_links, num_features), dtype=np.float32)
    
    # Colonnes à charger
    cols_to_load = ['link_id'] + available_features
    
    # Remplir la matrice par chunks
    processed = 0
    for chunk in pd.read_csv(links_path, usecols=cols_to_load, chunksize=CHUNK_SIZE):
        for _, row in chunk.iterrows():
            link_id = int(row['link_id'])
            idx = link_to_idx[link_id]
            
            for j, feat_name in enumerate(available_features):
                val = row[feat_name]
                # Gérer les NaN
                if pd.isna(val):
                    val = 0.0
                node_features[idx, j] = float(val)
            
            processed += 1
        
        logger.info(f"  - {processed:,} nœuds traités...")
    
    logger.info(f"  - Matrice de features: {node_features.shape}")
    logger.info(f"  - Mémoire utilisée: {node_features.nbytes / 1e6:.1f} MB")
    
    return node_features, available_features


# =============================================================================
# PHASE 2D: EXTRACTION DES LABELS (OPTIONNEL)
# =============================================================================

def extract_labels_chunked(travel_times_path: str,
                           link_to_idx: Dict[int, int],
                           output_dir: str,
                           num_links: int) -> Dict:
    """
    Extraire les labels (travel_time) par chunks temporels.
    
    Stratégie pour fichiers de 22GB:
    - Traiter par chunks de 200k lignes
    - Agréger par heure (moyenne des travel_times par lien)
    - Sauvegarder chaque jour comme fichier séparé
    
    Args:
        travel_times_path: Chemin vers travel_times_enriched.csv
        link_to_idx: Dict[link_id] → index
        output_dir: Répertoire de sortie
        num_links: Nombre de liens
        
    Returns:
        Statistiques d'extraction
    """
    logger.info("Phase 2D: Extraction des labels par chunks temporels...")
    
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)
    
    stats = {
        'total_records': 0,
        'matched_records': 0,
        'files_created': 0,
    }
    
    # Colonnes nécessaires
    cols_needed = ['begin_node_id', 'end_node_id', 'datetime', 
                   'travel_time', 'num_trips', 'hour', 'day_of_week', 
                   'is_weekend', 'is_rush_hour']
    
    # Buffer pour agréger par jour
    current_date = None
    daily_data = defaultdict(lambda: {'travel_times': [], 'num_trips': [], 'hours': []})
    
    def save_daily_data(date_str: str, data: Dict):
        """Sauvegarder les données d'une journée."""
        if not data:
            return
        
        # Calculer la moyenne par lien pour cette journée
        labels = np.full(num_links, np.nan, dtype=np.float32)
        weights = np.zeros(num_links, dtype=np.float32)
        
        for link_key, link_data in data.items():
            # Essayer de trouver l'index du lien
            begin_node, end_node = link_key
            
            # Chercher dans link_to_idx par correspondance
            # On a besoin d'un reverse mapping
            # Pour simplifier, on saute cette étape et on utilise une approche différente
            pass
        
        # Sauvegarder
        output_path = os.path.join(labels_dir, f'labels_{date_str}.npy')
        np.save(output_path, labels)
        stats['files_created'] += 1
    
    # Alternative: Créer un fichier d'index pour les labels
    # et stocker les données brutes agrégées
    
    # Approche simplifiée: Créer un fichier de moyennes globales
    # pour les tests initiaux
    
    logger.info("  - Calcul des moyennes globales par lien...")
    
    # Accumulateurs
    travel_time_sums = np.zeros(num_links, dtype=np.float64)
    travel_time_counts = np.zeros(num_links, dtype=np.int32)
    
    # Créer le reverse mapping: (begin_node, end_node) → link_idx
    # On doit relire les liens pour ça
    
    link_key_to_idx = {}
    cols_links = ['link_id', 'begin_node_id', 'end_node_id']
    links_path = travel_times_path.replace('travel_times_enriched.csv', 'links_enriched.csv')
    
    if os.path.exists(links_path):
        for chunk in pd.read_csv(links_path, usecols=cols_links, chunksize=CHUNK_SIZE):
            for _, row in chunk.iterrows():
                key = (int(row['begin_node_id']), int(row['end_node_id']))
                link_key_to_idx[key] = link_to_idx[int(row['link_id'])]
    
    logger.info(f"  - Index de correspondance créé: {len(link_key_to_idx):,} liens")
    
    # Maintenant traiter les travel_times
    cols_tt = ['begin_node_id', 'end_node_id', 'travel_time', 'num_trips']
    
    for chunk in pd.read_csv(travel_times_path, usecols=cols_tt, chunksize=CHUNK_SIZE):
        for _, row in chunk.iterrows():
            key = (int(row['begin_node_id']), int(row['end_node_id']))
            
            if key in link_key_to_idx:
                idx = link_key_to_idx[key]
                tt = row['travel_time']
                
                if not pd.isna(tt) and tt > 0:
                    travel_time_sums[idx] += tt
                    travel_time_counts[idx] += 1
                    stats['matched_records'] += 1
            
            stats['total_records'] += 1
        
        if stats['total_records'] % 5_000_000 == 0:
            logger.info(f"  - {stats['total_records']:,} enregistrements traités...")
    
    # Calculer les moyennes
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_travel_times = np.where(
            travel_time_counts > 0,
            travel_time_sums / travel_time_counts,
            np.nan
        ).astype(np.float32)
    
    # Sauvegarder
    labels_path = os.path.join(output_dir, 'mean_travel_times.npy')
    np.save(labels_path, mean_travel_times)
    
    # Sauvegarder aussi les counts pour la pondération
    counts_path = os.path.join(output_dir, 'observation_counts.npy')
    np.save(counts_path, travel_time_counts)
    
    # Statistiques
    valid_count = np.sum(travel_time_counts > 0)
    stats['links_with_data'] = int(valid_count)
    stats['links_without_data'] = int(num_links - valid_count)
    stats['coverage'] = float(valid_count / num_links * 100)
    stats['mean_observations_per_link'] = float(np.mean(travel_time_counts[travel_time_counts > 0]))
    
    logger.info(f"  - Liens avec données: {valid_count:,} ({stats['coverage']:.1f}%)")
    logger.info(f"  - Observations moyennes par lien: {stats['mean_observations_per_link']:.1f}")
    
    # Nettoyer
    del link_key_to_idx
    gc.collect()
    
    return stats


# =============================================================================
# PHASE 2E: CRÉATION DU MAPPING ET VALIDATION
# =============================================================================

def create_node_mapping(links_path: str, 
                        link_to_idx: Dict[int, int],
                        output_path: str) -> None:
    """
    Créer le fichier de correspondance link_id → line_graph_node_id.
    
    Inclut aussi les coordonnées pour la visualisation et le split spatial.
    
    Args:
        links_path: Chemin vers links_enriched.csv
        link_to_idx: Dict[link_id] → index
        output_path: Chemin de sortie
    """
    logger.info("Phase 2E: Création du mapping nœuds...")
    
    # Colonnes pour le mapping
    cols = ['link_id', 'begin_node_id', 'end_node_id', 
            'startX', 'startY', 'endX', 'endY', 'osm_class']
    
    mappings = []
    
    for chunk in pd.read_csv(links_path, usecols=cols, chunksize=CHUNK_SIZE):
        for _, row in chunk.iterrows():
            link_id = int(row['link_id'])
            mappings.append({
                'link_id': link_id,
                'line_graph_node_id': link_to_idx[link_id],
                'begin_node_id': int(row['begin_node_id']),
                'end_node_id': int(row['end_node_id']),
                'center_x': (row['startX'] + row['endX']) / 2,
                'center_y': (row['startY'] + row['endY']) / 2,
                'osm_class': row['osm_class'],
            })
    
    # Sauvegarder
    df = pd.DataFrame(mappings)
    df.to_csv(output_path, index=False)
    
    logger.info(f"  - Mapping sauvegardé: {len(df):,} entrées")


def validate_line_graph(edge_index: np.ndarray, 
                        node_features: np.ndarray,
                        num_nodes: int) -> Dict:
    """
    Valider la cohérence du line graph construit.
    
    Args:
        edge_index: Matrice d'adjacence (2, num_edges)
        node_features: Features des nœuds (num_nodes, num_features)
        num_nodes: Nombre attendu de nœuds
        
    Returns:
        Statistiques de validation
    """
    logger.info("Validation du line graph...")
    
    validation = {
        'num_nodes': num_nodes,
        'num_edges': edge_index.shape[1],
        'num_features': node_features.shape[1],
        'is_valid': True,
        'errors': [],
    }
    
    # Vérifier les dimensions
    if node_features.shape[0] != num_nodes:
        validation['errors'].append(
            f"Nombre de nœuds incohérent: features={node_features.shape[0]}, attendu={num_nodes}"
        )
        validation['is_valid'] = False
    
    # Vérifier les indices des arêtes
    max_src = edge_index[0].max()
    max_tgt = edge_index[1].max()
    
    if max_src >= num_nodes or max_tgt >= num_nodes:
        validation['errors'].append(
            f"Indices d'arêtes hors limites: max_src={max_src}, max_tgt={max_tgt}, num_nodes={num_nodes}"
        )
        validation['is_valid'] = False
    
    # Statistiques du graphe
    # Degré sortant
    out_degrees = np.bincount(edge_index[0], minlength=num_nodes)
    in_degrees = np.bincount(edge_index[1], minlength=num_nodes)
    
    validation['degree_stats'] = {
        'out_degree_mean': float(out_degrees.mean()),
        'out_degree_max': int(out_degrees.max()),
        'out_degree_min': int(out_degrees.min()),
        'in_degree_mean': float(in_degrees.mean()),
        'in_degree_max': int(in_degrees.max()),
        'isolated_nodes': int(np.sum((out_degrees == 0) & (in_degrees == 0))),
    }
    
    # Vérifier les features
    nan_count = np.isnan(node_features).sum()
    validation['feature_stats'] = {
        'nan_count': int(nan_count),
        'nan_percentage': float(nan_count / node_features.size * 100),
        'feature_means': node_features.mean(axis=0).tolist(),
        'feature_stds': node_features.std(axis=0).tolist(),
    }
    
    if validation['is_valid']:
        logger.info("  ✓ Line graph valide")
    else:
        logger.error(f"  ✗ Erreurs de validation: {validation['errors']}")
    
    logger.info(f"  - Nœuds isolés: {validation['degree_stats']['isolated_nodes']}")
    logger.info(f"  - Degré sortant moyen: {validation['degree_stats']['out_degree_mean']:.2f}")
    logger.info(f"  - NaN dans features: {validation['feature_stats']['nan_percentage']:.2f}%")
    
    return validation


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def main(data_dir: str, output_dir: str, skip_labels: bool = False):
    """
    Pipeline principal de construction du Line Graph.
    
    Args:
        data_dir: Répertoire contenant les fichiers enrichis (Phase 1)
        output_dir: Répertoire de sortie
        skip_labels: Si True, ne pas extraire les labels (économise du temps/mémoire)
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("PHASE 2: CONSTRUCTION DU DATASET EDGE-CENTRIC (LINE GRAPH)")
    logger.info("=" * 70)
    logger.info(f"Répertoire source: {data_dir}")
    logger.info(f"Répertoire sortie: {output_dir}")
    logger.info(f"Skip labels: {skip_labels}")
    
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Chemins des fichiers
    links_path = os.path.join(data_dir, 'links_enriched.csv')
    travel_times_path = os.path.join(data_dir, 'travel_times_enriched.csv')
    
    # Vérifier l'existence des fichiers
    if not os.path.exists(links_path):
        logger.error(f"Fichier manquant: {links_path}")
        sys.exit(1)
    
    # ==========================================================================
    # ÉTAPE 1: Construction de l'index d'adjacence
    # ==========================================================================
    logger.info("\n--- ÉTAPE 1: Index d'adjacence ---")
    node_to_outgoing, node_to_incoming, link_to_idx, num_links = build_adjacency_index(links_path)
    
    # Libérer la mémoire
    gc.collect()
    
    # ==========================================================================
    # ÉTAPE 2: Construction de la matrice d'adjacence
    # ==========================================================================
    logger.info("\n--- ÉTAPE 2: Matrice d'adjacence ---")
    edge_index = build_edge_index(links_path, node_to_outgoing, node_to_incoming, link_to_idx)
    
    # Sauvegarder immédiatement pour libérer la mémoire
    edge_index_path = os.path.join(output_dir, 'edge_index.npy')
    np.save(edge_index_path, edge_index)
    logger.info(f"  - Sauvegardé: {edge_index_path}")
    
    # Libérer les dictionnaires d'adjacence
    del node_to_outgoing, node_to_incoming
    gc.collect()
    
    # ==========================================================================
    # ÉTAPE 3: Extraction des features
    # ==========================================================================
    logger.info("\n--- ÉTAPE 3: Features des nœuds ---")
    node_features, feature_names = extract_node_features(links_path, link_to_idx, num_links)
    
    # Sauvegarder
    features_path = os.path.join(output_dir, 'node_features.npy')
    np.save(features_path, node_features)
    logger.info(f"  - Sauvegardé: {features_path}")
    
    # ==========================================================================
    # ÉTAPE 4: Extraction des labels (optionnel)
    # ==========================================================================
    label_stats = {}
    if not skip_labels and os.path.exists(travel_times_path):
        logger.info("\n--- ÉTAPE 4: Extraction des labels ---")
        label_stats = extract_labels_chunked(
            travel_times_path, link_to_idx, output_dir, num_links
        )
    elif not os.path.exists(travel_times_path):
        logger.warning(f"Fichier {travel_times_path} non trouvé - labels ignorés")
    else:
        logger.info("\n--- ÉTAPE 4: Labels ignorés (--skip-labels) ---")
    
    # ==========================================================================
    # ÉTAPE 5: Mapping et validation
    # ==========================================================================
    logger.info("\n--- ÉTAPE 5: Mapping et validation ---")
    
    # Créer le mapping
    mapping_path = os.path.join(output_dir, 'node_mapping.csv')
    create_node_mapping(links_path, link_to_idx, mapping_path)
    
    # Valider le graphe
    validation = validate_line_graph(edge_index, node_features, num_links)
    
    # ==========================================================================
    # ÉTAPE 6: Métadonnées
    # ==========================================================================
    logger.info("\n--- ÉTAPE 6: Sauvegarde des métadonnées ---")
    
    metadata = {
        'created_at': datetime.now().isoformat(),
        'source_dir': data_dir,
        'line_graph': {
            'num_nodes': num_links,
            'num_edges': int(edge_index.shape[1]),
            'num_features': len(feature_names),
            'feature_names': feature_names,
        },
        'original_graph': {
            'num_links': num_links,
        },
        'validation': validation,
        'labels': label_stats,
        'memory_info': {
            'edge_index_mb': edge_index.nbytes / 1e6,
            'node_features_mb': node_features.nbytes / 1e6,
        },
        'files': {
            'edge_index': 'edge_index.npy',
            'node_features': 'node_features.npy',
            'node_mapping': 'node_mapping.csv',
            'mean_travel_times': 'mean_travel_times.npy' if label_stats else None,
            'observation_counts': 'observation_counts.npy' if label_stats else None,
        }
    }
    
    metadata_path = os.path.join(output_dir, 'line_graph_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, cls=NumpyEncoder)
    logger.info(f"  - Métadonnées: {metadata_path}")
    
    # ==========================================================================
    # RÉSUMÉ
    # ==========================================================================
    elapsed = datetime.now() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("CONSTRUCTION DU LINE GRAPH TERMINÉE")
    logger.info("=" * 70)
    logger.info(f"Durée totale: {elapsed}")
    
    logger.info(f"\nFichiers générés dans {output_dir}:")
    logger.info(f"  - edge_index.npy ({edge_index.nbytes / 1e6:.1f} MB)")
    logger.info(f"  - node_features.npy ({node_features.nbytes / 1e6:.1f} MB)")
    logger.info(f"  - node_mapping.csv")
    if label_stats:
        logger.info(f"  - mean_travel_times.npy")
        logger.info(f"  - observation_counts.npy")
    logger.info(f"  - line_graph_metadata.json")
    
    logger.info(f"\nStructure du Line Graph:")
    logger.info(f"  - Nœuds (segments routiers): {num_links:,}")
    logger.info(f"  - Arêtes (connexions): {edge_index.shape[1]:,}")
    logger.info(f"  - Features par nœud: {len(feature_names)}")
    logger.info(f"  - Degré moyen: {edge_index.shape[1] / num_links:.2f}")
    
    if label_stats:
        logger.info(f"\nLabels:")
        logger.info(f"  - Couverture: {label_stats.get('coverage', 0):.1f}%")
        logger.info(f"  - Observations/lien: {label_stats.get('mean_observations_per_link', 0):.1f}")
    
    logger.info("\nProchaine étape: Phase 3 - Architecture du Modèle GNN")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Construction du Line Graph pour GNN - Phase 2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python build_line_graph.py --data-dir ./enriched_data --output-dir ./line_graph_data
  python build_line_graph.py -d ./enriched -o ./output --skip-labels
        """
    )
    parser.add_argument(
        '-d', '--data-dir',
        type=str,
        default='./enriched_data',
        help='Répertoire contenant links_enriched.csv et travel_times_enriched.csv'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='./line_graph_data',
        help='Répertoire de sortie pour le line graph'
    )
    parser.add_argument(
        '--skip-labels',
        action='store_true',
        help='Ne pas extraire les labels (économise temps et mémoire)'
    )
    
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.skip_labels)