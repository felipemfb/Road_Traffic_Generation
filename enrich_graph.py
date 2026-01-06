#!/usr/bin/env python3
"""
enrich_graph.py - Phase 1: Enrichissement et Normalisation du Graphe Routier
=============================================================================

Script de Feature Engineering pour le projet NYC Traffic Prediction.
Génère des attributs relatifs et normalisés pour garantir la transférabilité
du modèle vers d'autres villes (ex: Marseille) après entraînement sur NYC.

Optimisé pour traiter des fichiers volumineux (5GB+) via:
- Traitement par chunks (pandas)
- Types de données optimisés (catégories, float32)
- Calculs vectorisés (numpy)

Usage:
    python enrich_graph.py --data-dir ./travel_times_2013 --output-dir ./enriched_data

Entrées requises:
    - nodes.csv: Nœuds du graphe (intersections)
    - links.csv: Arêtes du graphe (segments routiers)  
    - travel_times_2013.csv: Données de trafic temporelles

Sorties:
    - links_enriched.csv: Liens enrichis avec features normalisées
    - nodes_enriched.csv: Nœuds enrichis avec métriques de centralité
    - enrichment_stats.json: Statistiques de normalisation pour inférence
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import dijkstra


class NumpyEncoder(json.JSONEncoder):
    """Encodeur JSON personnalisé pour les types numpy."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Taille des chunks pour le traitement des gros fichiers
CHUNK_SIZE = 500_000

# Types de données optimisés pour réduire l'empreinte mémoire
# Note: Les colonnes avec valeurs booléennes textuelles (true/false) ou 
# valeurs manquantes sont chargées en 'object' puis converties manuellement
NODES_DTYPES = {
    'node_id': 'int64',
    'is_complete': 'object',  # Peut être 't', 'f', 'true', 'false', 0, 1
    'num_in_links': 'float32',  # float pour gérer les NaN
    'num_out_links': 'float32',
    'osm_traffic_controller': 'object',  # Peut contenir des NaN
    'xcoord': 'float64',
    'ycoord': 'float64',
    'grid_region_id': 'float32'  # float pour gérer les NaN
}

LINKS_DTYPES = {
    'link_id': 'int64',
    'begin_node_id': 'int64',
    'end_node_id': 'int64',
    'begin_angle': 'float32',
    'end_angle': 'float32',
    'street_length': 'float32',
    'osm_name': 'object',  # Peut contenir des NaN
    'osm_class': 'object',  # Converti en category après chargement
    'osm_way_id': 'float64',  # float pour gérer les NaN
    'startX': 'float64',
    'startY': 'float64',
    'endX': 'float64',
    'endY': 'float64'
}

TRAVEL_TIMES_DTYPES = {
    'begin_node_id': 'int64',
    'end_node_id': 'int64',
    'travel_time': 'float32',
    'num_trips': 'int32'
}

# Mapping OSM vers catégories universelles (indépendant du pays)
OSM_CLASS_HIERARCHY = {
    # Artères principales (haute capacité)
    'motorway': 1,
    'motorway_link': 1,
    'trunk': 1,
    'trunk_link': 1,
    # Routes principales
    'primary': 2,
    'primary_link': 2,
    # Routes secondaires
    'secondary': 3,
    'secondary_link': 3,
    # Routes tertiaires
    'tertiary': 4,
    'tertiary_link': 4,
    # Routes résidentielles et de service
    'residential': 5,
    'living_street': 5,
    'unclassified': 5,
    'service': 6,
}

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('enrich_graph.log')
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def estimate_city_center(nodes_df: pd.DataFrame) -> Tuple[float, float]:
    """
    Estime le centre de la ville comme le centroïde pondéré par la densité.
    
    Utilise une approche robuste basée sur la médiane pour éviter
    l'influence des outliers géographiques.
    
    Args:
        nodes_df: DataFrame des nœuds avec colonnes xcoord, ycoord
        
    Returns:
        Tuple (center_x, center_y) représentant le centre estimé
    """
    # Approche 1: Médiane (robuste aux outliers)
    center_x = nodes_df['xcoord'].median()
    center_y = nodes_df['ycoord'].median()
    
    logger.info(f"Centre ville estimé: ({center_x:.6f}, {center_y:.6f})")
    return center_x, center_y


def clean_nodes_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie et convertit les types du DataFrame des nœuds.
    
    Gère les colonnes booléennes textuelles ('t'/'f', 'true'/'false')
    et remplit les valeurs manquantes.
    """
    # Conversion de is_complete (peut être 't', 'f', 'true', 'false', 0, 1)
    if 'is_complete' in df.columns:
        df['is_complete'] = df['is_complete'].astype(str).str.lower()
        df['is_complete'] = df['is_complete'].map({
            't': 1, 'true': 1, '1': 1, '1.0': 1,
            'f': 0, 'false': 0, '0': 0, '0.0': 0,
            'nan': 0, 'none': 0, '': 0
        }).fillna(0).astype('int8')
    
    # Remplir les NaN pour les colonnes numériques
    if 'num_in_links' in df.columns:
        df['num_in_links'] = df['num_in_links'].fillna(0).astype('int16')
    if 'num_out_links' in df.columns:
        df['num_out_links'] = df['num_out_links'].fillna(0).astype('int16')
    if 'grid_region_id' in df.columns:
        df['grid_region_id'] = df['grid_region_id'].fillna(-1).astype('int32')
    
    # Convertir osm_traffic_controller en category
    if 'osm_traffic_controller' in df.columns:
        df['osm_traffic_controller'] = df['osm_traffic_controller'].fillna('unknown').astype('category')
    
    return df


def clean_links_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie et convertit les types du DataFrame des liens.
    """
    # Convertir osm_class en category
    if 'osm_class' in df.columns:
        df['osm_class'] = df['osm_class'].fillna('unclassified').astype('category')
    
    # Remplir les NaN pour osm_name
    if 'osm_name' in df.columns:
        df['osm_name'] = df['osm_name'].fillna('').astype('category')
    
    # Remplir les NaN pour les colonnes numériques
    if 'street_length' in df.columns:
        df['street_length'] = df['street_length'].fillna(df['street_length'].median())
    
    return df


def haversine_distance(lon1: np.ndarray, lat1: np.ndarray, 
                       lon2: float, lat2: float) -> np.ndarray:
    """
    Calcule la distance haversine (en km) de manière vectorisée.
    
    Args:
        lon1, lat1: Coordonnées des points (arrays)
        lon2, lat2: Coordonnées du point de référence (scalaires)
        
    Returns:
        Array des distances en kilomètres
    """
    R = 6371  # Rayon de la Terre en km
    
    lon1_rad = np.radians(lon1)
    lat1_rad = np.radians(lat1)
    lon2_rad = np.radians(lon2)
    lat2_rad = np.radians(lat2)
    
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def encode_cyclic(values: np.ndarray, period: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode des valeurs cycliques en composantes sin/cos.
    
    Permet au modèle de comprendre que 23h est proche de 0h,
    ou que décembre est proche de janvier.
    
    Args:
        values: Valeurs à encoder
        period: Période du cycle (24 pour heures, 7 pour jours, etc.)
        
    Returns:
        Tuple (sin_component, cos_component)
    """
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def compute_z_score(values: pd.Series, stats: Optional[Dict] = None) -> Tuple[pd.Series, Dict]:
    """
    Calcule le Z-score avec gestion des valeurs manquantes.
    
    Args:
        values: Série de valeurs à normaliser
        stats: Dictionnaire optionnel avec 'mean' et 'std' pré-calculés
        
    Returns:
        Tuple (valeurs_normalisées, statistiques)
    """
    if stats is None:
        mean = values.mean()
        std = values.std()
        # Éviter division par zéro
        std = std if std > 1e-10 else 1.0
        stats = {'mean': float(mean), 'std': float(std)}
    
    normalized = (values - stats['mean']) / stats['std']
    return normalized, stats


def compute_percentile_rank(values: pd.Series) -> pd.Series:
    """
    Calcule le rang percentile (0 à 1) pour chaque valeur.
    
    Transformation uniforme qui préserve l'ordre mais normalise
    la distribution, utile pour les features très asymétriques.
    
    Args:
        values: Série de valeurs
        
    Returns:
        Série des rangs percentiles
    """
    return values.rank(pct=True)


# =============================================================================
# ENRICHISSEMENT DES NŒUDS
# =============================================================================

def enrich_nodes(nodes_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Enrichit les nœuds avec des features de centralité relative.
    
    Features générées:
    - distance_to_center_km: Distance au centre (absolu, pour debug)
    - centrality_percentile: Rang percentile (0=centre, 1=périphérie)
    - degree_in_normalized: Degré entrant normalisé (Z-score)
    - degree_out_normalized: Degré sortant normalisé (Z-score)
    - degree_total_percentile: Degré total en percentile
    
    Args:
        nodes_df: DataFrame des nœuds bruts
        
    Returns:
        Tuple (DataFrame enrichi, statistiques de normalisation)
    """
    logger.info("Enrichissement des nœuds...")
    df = nodes_df.copy()
    stats = {}
    
    # 1. Calcul de la distance au centre
    center_x, center_y = estimate_city_center(df)
    stats['city_center'] = {'x': center_x, 'y': center_y}
    
    df['distance_to_center_km'] = haversine_distance(
        df['xcoord'].values, df['ycoord'].values,
        center_x, center_y
    )
    
    # 2. Centralité relative (percentile)
    # 0 = centre absolu, 1 = périphérie extrême
    df['centrality_percentile'] = compute_percentile_rank(df['distance_to_center_km'])
    
    # 3. Normalisation des degrés (Z-score)
    df['degree_in_normalized'], stats['degree_in'] = compute_z_score(
        df['num_in_links'].astype(float)
    )
    df['degree_out_normalized'], stats['degree_out'] = compute_z_score(
        df['num_out_links'].astype(float)
    )
    
    # 4. Degré total en percentile
    df['degree_total'] = df['num_in_links'] + df['num_out_links']
    df['degree_total_percentile'] = compute_percentile_rank(df['degree_total'])
    
    # 5. Indicateur de contrôleur de trafic (one-hot simplifié)
    df['has_traffic_signal'] = (
        df['osm_traffic_controller'].astype(str).str.contains('signal', case=False, na=False)
    ).astype('int8')
    
    logger.info(f"  - {len(df)} nœuds enrichis")
    logger.info(f"  - Distance max au centre: {df['distance_to_center_km'].max():.2f} km")
    
    return df, stats


# =============================================================================
# ENRICHISSEMENT DES LIENS
# =============================================================================

def enrich_links(links_df: pd.DataFrame, nodes_enriched: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Enrichit les liens avec des features topologiques et contextuelles normalisées.
    
    Features générées:
    - road_hierarchy: Classification hiérarchique universelle (1-6)
    - length_percentile: Longueur relative en percentile
    - length_zscore: Longueur normalisée (Z-score)
    - angle_change: Changement d'angle (courbure du segment)
    - angle_change_sin/cos: Encodage cyclique de l'angle
    - start_centrality: Centralité du nœud de départ
    - end_centrality: Centralité du nœud d'arrivée
    - avg_centrality: Centralité moyenne du segment
    - connects_center: Indicateur si le segment est proche du centre
    
    Args:
        links_df: DataFrame des liens bruts
        nodes_enriched: DataFrame des nœuds enrichis
        
    Returns:
        Tuple (DataFrame enrichi, statistiques de normalisation)
    """
    logger.info("Enrichissement des liens...")
    df = links_df.copy()
    stats = {}
    
    # 1. Classification hiérarchique universelle
    df['road_hierarchy'] = df['osm_class'].map(OSM_CLASS_HIERARCHY).fillna(6).astype('int8')
    
    # Statistiques de distribution
    hierarchy_dist = df['road_hierarchy'].value_counts(normalize=True).sort_index()
    stats['road_hierarchy_distribution'] = hierarchy_dist.to_dict()
    logger.info(f"  - Distribution hiérarchie: {dict(hierarchy_dist.round(3))}")
    
    # 2. Longueur relative
    df['length_percentile'] = compute_percentile_rank(df['street_length'])
    df['length_zscore'], stats['street_length'] = compute_z_score(df['street_length'])
    
    # 3. Géométrie: changement d'angle (courbure)
    df['angle_change'] = (df['end_angle'] - df['begin_angle']).abs()
    # Normaliser à [0, 180] pour les angles
    df['angle_change'] = df['angle_change'].apply(lambda x: min(x, 360 - x) if x > 180 else x)
    
    # Encodage cyclique de l'angle de départ (orientation du segment)
    df['begin_angle_sin'], df['begin_angle_cos'] = encode_cyclic(df['begin_angle'].values, 360)
    
    # 4. Jointure avec les features des nœuds
    node_features = nodes_enriched[['node_id', 'centrality_percentile', 'degree_total_percentile']].copy()
    
    # Features du nœud de départ
    df = df.merge(
        node_features.rename(columns={
            'centrality_percentile': 'start_centrality',
            'degree_total_percentile': 'start_degree_percentile'
        }),
        left_on='begin_node_id',
        right_on='node_id',
        how='left'
    ).drop(columns=['node_id'])
    
    # Features du nœud d'arrivée
    df = df.merge(
        node_features.rename(columns={
            'centrality_percentile': 'end_centrality',
            'degree_total_percentile': 'end_degree_percentile'
        }),
        left_on='end_node_id',
        right_on='node_id',
        how='left'
    ).drop(columns=['node_id'])
    
    # 5. Features agrégées du segment
    df['avg_centrality'] = (df['start_centrality'] + df['end_centrality']) / 2
    df['centrality_gradient'] = df['end_centrality'] - df['start_centrality']
    
    # Indicateur binaire: segment proche du centre (premier quartile)
    df['connects_center'] = (df['avg_centrality'] < 0.25).astype('int8')
    
    # 6. Estimation de la capacité relative basée sur la hiérarchie
    # Mapping approximatif: hiérarchie -> capacité relative
    capacity_map = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.25, 6: 0.15}
    df['estimated_capacity'] = df['road_hierarchy'].map(capacity_map).astype('float32')
    
    logger.info(f"  - {len(df)} liens enrichis")
    
    return df, stats


# =============================================================================
# ENRICHISSEMENT TEMPOREL DES DONNÉES DE TRAFIC
# =============================================================================

def process_travel_times_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Traite un chunk de données de temps de trajet.
    
    Ajoute les features temporelles cycliques:
    - hour_sin/cos: Encodage de l'heure
    - day_of_week_sin/cos: Encodage du jour
    - is_weekend: Indicateur binaire
    - is_rush_hour: Indicateur des heures de pointe
    
    Args:
        chunk: DataFrame chunk avec colonne 'datetime'
        
    Returns:
        DataFrame enrichi avec features temporelles
    """
    df = chunk.copy()
    
    # Parser la datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Extraire les composantes
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    
    # Encodages cycliques
    df['hour_sin'], df['hour_cos'] = encode_cyclic(df['hour'].values, 24)
    df['day_sin'], df['day_cos'] = encode_cyclic(df['day_of_week'].values, 7)
    df['month_sin'], df['month_cos'] = encode_cyclic(df['month'].values, 12)
    
    # Indicateurs binaires
    df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
    
    # Heures de pointe: 7-9h et 17-19h en semaine
    df['is_rush_hour'] = (
        (~df['is_weekend'].astype(bool)) & 
        (((df['hour'] >= 7) & (df['hour'] <= 9)) | 
         ((df['hour'] >= 17) & (df['hour'] <= 19)))
    ).astype('int8')
    
    # Période de la journée (catégorielle ordonnée)
    conditions = [
        (df['hour'] >= 6) & (df['hour'] < 10),   # Matin
        (df['hour'] >= 10) & (df['hour'] < 16),  # Journée
        (df['hour'] >= 16) & (df['hour'] < 20),  # Soir
        (df['hour'] >= 20) | (df['hour'] < 6),   # Nuit
    ]
    choices = [1, 2, 3, 4]
    df['time_period'] = np.select(conditions, choices, default=2).astype('int8')
    
    return df


def enrich_travel_times(input_path: str, output_path: str, links_enriched: pd.DataFrame) -> Dict:
    """
    Enrichit les données de temps de trajet par chunks pour gérer les gros fichiers.
    
    Args:
        input_path: Chemin vers travel_times_2013.csv
        output_path: Chemin de sortie
        links_enriched: DataFrame des liens enrichis pour la jointure
        
    Returns:
        Statistiques d'enrichissement
    """
    logger.info(f"Enrichissement des temps de trajet (par chunks de {CHUNK_SIZE})...")
    
    # Préparer les features des liens pour la jointure
    link_features = links_enriched[[
        'begin_node_id', 'end_node_id',
        'road_hierarchy', 'length_percentile', 'avg_centrality',
        'estimated_capacity', 'connects_center'
    ]].copy()
    
    # Créer une clé composite pour la jointure
    link_features['link_key'] = (
        link_features['begin_node_id'].astype(str) + '_' + 
        link_features['end_node_id'].astype(str)
    )
    link_features = link_features.drop(columns=['begin_node_id', 'end_node_id'])
    link_features_dict = link_features.set_index('link_key').to_dict('index')
    
    stats = {
        'total_records': 0,
        'matched_records': 0,
        'chunks_processed': 0
    }
    
    # Traitement par chunks
    first_chunk = True
    for chunk in pd.read_csv(input_path, dtype=TRAVEL_TIMES_DTYPES, chunksize=CHUNK_SIZE):
        # Enrichir les features temporelles
        chunk = process_travel_times_chunk(chunk)
        
        # Créer la clé de jointure
        chunk['link_key'] = (
            chunk['begin_node_id'].astype(str) + '_' + 
            chunk['end_node_id'].astype(str)
        )
        
        # Jointure avec les features des liens
        for col in ['road_hierarchy', 'length_percentile', 'avg_centrality', 
                    'estimated_capacity', 'connects_center']:
            chunk[col] = chunk['link_key'].map(
                lambda k: link_features_dict.get(k, {}).get(col, np.nan)
            )
        
        # Supprimer la clé temporaire
        chunk = chunk.drop(columns=['link_key'])
        
        # Statistiques
        stats['total_records'] += len(chunk)
        stats['matched_records'] += chunk['road_hierarchy'].notna().sum()
        stats['chunks_processed'] += 1
        
        # Écrire en mode append
        chunk.to_csv(
            output_path,
            mode='w' if first_chunk else 'a',
            header=first_chunk,
            index=False
        )
        first_chunk = False
        
        if stats['chunks_processed'] % 10 == 0:
            logger.info(f"  - {stats['total_records']:,} enregistrements traités...")
    
    match_rate = stats['matched_records'] / stats['total_records'] * 100 if stats['total_records'] > 0 else 0
    logger.info(f"  - Total: {stats['total_records']:,} enregistrements")
    logger.info(f"  - Taux de correspondance liens: {match_rate:.1f}%")
    
    return stats


# =============================================================================
# CALCUL DES FEATURES DE VOISINAGE (OPTIONNEL - COÛTEUX EN MÉMOIRE)
# =============================================================================

def compute_neighborhood_features(links_df: pd.DataFrame, 
                                   nodes_df: pd.DataFrame,
                                   k_hops: int = 2) -> pd.DataFrame:
    """
    Calcule des features agrégées du voisinage pour chaque lien.
    
    ATTENTION: Cette fonction peut être coûteuse en mémoire pour de gros graphes.
    À utiliser avec précaution sur des données de 5GB+.
    
    Features:
    - neighbor_avg_hierarchy: Hiérarchie moyenne des liens voisins
    - neighbor_avg_length: Longueur moyenne des liens voisins
    - local_density: Densité locale du réseau (nb liens dans le voisinage)
    
    Args:
        links_df: DataFrame des liens enrichis
        nodes_df: DataFrame des nœuds
        k_hops: Nombre de sauts pour définir le voisinage
        
    Returns:
        DataFrame avec features de voisinage ajoutées
    """
    logger.info(f"Calcul des features de voisinage ({k_hops}-hops)...")
    logger.warning("  - Cette opération peut être coûteuse en mémoire!")
    
    df = links_df.copy()
    
    # Construire le graphe d'adjacence des liens
    # Un lien A est voisin d'un lien B si end_node(A) == begin_node(B)
    
    # Index pour accès rapide
    end_to_links = df.groupby('end_node_id')['link_id'].apply(list).to_dict()
    begin_to_links = df.groupby('begin_node_id')['link_id'].apply(list).to_dict()
    link_to_idx = {lid: idx for idx, lid in enumerate(df['link_id'])}
    
    # Pour chaque lien, trouver les voisins sortants
    neighbor_hierarchies = []
    neighbor_lengths = []
    local_densities = []
    
    for idx, row in df.iterrows():
        end_node = row['end_node_id']
        
        # Liens qui partent du nœud d'arrivée (voisins sortants)
        outgoing = begin_to_links.get(end_node, [])
        
        if len(outgoing) > 0:
            neighbor_h = df[df['link_id'].isin(outgoing)]['road_hierarchy'].mean()
            neighbor_l = df[df['link_id'].isin(outgoing)]['street_length'].mean()
        else:
            neighbor_h = row['road_hierarchy']  # Fallback
            neighbor_l = row['street_length']
        
        neighbor_hierarchies.append(neighbor_h)
        neighbor_lengths.append(neighbor_l)
        local_densities.append(len(outgoing))
        
        if idx % 50000 == 0 and idx > 0:
            logger.info(f"  - {idx:,} liens traités...")
    
    df['neighbor_avg_hierarchy'] = neighbor_hierarchies
    df['neighbor_avg_length'] = neighbor_lengths
    df['local_out_degree'] = local_densities
    
    # Normaliser la densité locale
    df['local_density_percentile'] = compute_percentile_rank(pd.Series(local_densities))
    
    logger.info(f"  - Features de voisinage calculées pour {len(df)} liens")
    
    return df


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def main(data_dir: str, output_dir: str, compute_neighbors: bool = False):
    """
    Pipeline principal d'enrichissement du graphe.
    
    Args:
        data_dir: Répertoire contenant les fichiers sources
        output_dir: Répertoire de sortie
        compute_neighbors: Si True, calcule les features de voisinage (coûteux)
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("PHASE 1: ENRICHISSEMENT ET NORMALISATION DU GRAPHE")
    logger.info("=" * 60)
    logger.info(f"Répertoire source: {data_dir}")
    logger.info(f"Répertoire sortie: {output_dir}")
    
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Chemins des fichiers
    nodes_path = os.path.join(data_dir, 'nodes.csv')
    links_path = os.path.join(data_dir, 'links.csv')
    travel_times_path = os.path.join(data_dir, 'travel_times_2013.csv')
    
    # Vérifier l'existence des fichiers
    for path, name in [(nodes_path, 'nodes.csv'), (links_path, 'links.csv')]:
        if not os.path.exists(path):
            logger.error(f"Fichier manquant: {path}")
            sys.exit(1)
    
    # ==========================================================================
    # ÉTAPE 1: Charger et enrichir les nœuds
    # ==========================================================================
    logger.info("\n--- ÉTAPE 1: Chargement des nœuds ---")
    nodes_df = pd.read_csv(nodes_path, dtype=NODES_DTYPES)
    logger.info(f"  - {len(nodes_df):,} nœuds chargés")
    
    # Nettoyage des données
    nodes_df = clean_nodes_dataframe(nodes_df)
    logger.info(f"  - Données nettoyées et types convertis")
    
    nodes_enriched, nodes_stats = enrich_nodes(nodes_df)
    
    # ==========================================================================
    # ÉTAPE 2: Charger et enrichir les liens
    # ==========================================================================
    logger.info("\n--- ÉTAPE 2: Chargement des liens ---")
    links_df = pd.read_csv(links_path, dtype=LINKS_DTYPES)
    logger.info(f"  - {len(links_df):,} liens chargés")
    
    # Nettoyage des données
    links_df = clean_links_dataframe(links_df)
    logger.info(f"  - Données nettoyées et types convertis")
    
    links_enriched, links_stats = enrich_links(links_df, nodes_enriched)
    
    # ==========================================================================
    # ÉTAPE 2b (OPTIONNEL): Features de voisinage
    # ==========================================================================
    if compute_neighbors:
        logger.info("\n--- ÉTAPE 2b: Features de voisinage ---")
        links_enriched = compute_neighborhood_features(links_enriched, nodes_enriched)
    
    # ==========================================================================
    # ÉTAPE 3: Enrichir les temps de trajet (si le fichier existe)
    # ==========================================================================
    travel_stats = {}
    if os.path.exists(travel_times_path):
        logger.info("\n--- ÉTAPE 3: Enrichissement des temps de trajet ---")
        travel_output = os.path.join(output_dir, 'travel_times_enriched.csv')
        travel_stats = enrich_travel_times(travel_times_path, travel_output, links_enriched)
    else:
        logger.warning(f"Fichier {travel_times_path} non trouvé - étape ignorée")
    
    # ==========================================================================
    # ÉTAPE 4: Sauvegarde des résultats
    # ==========================================================================
    logger.info("\n--- ÉTAPE 4: Sauvegarde ---")
    
    # Sauvegarder les nœuds enrichis
    nodes_output = os.path.join(output_dir, 'nodes_enriched.csv')
    nodes_enriched.to_csv(nodes_output, index=False)
    logger.info(f"  - Nœuds enrichis: {nodes_output}")
    
    # Sauvegarder les liens enrichis
    links_output = os.path.join(output_dir, 'links_enriched.csv')
    links_enriched.to_csv(links_output, index=False)
    logger.info(f"  - Liens enrichis: {links_output}")
    
    # Sauvegarder les statistiques de normalisation (pour l'inférence)
    all_stats = {
        'nodes': nodes_stats,
        'links': links_stats,
        'travel_times': travel_stats,
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'source_dir': data_dir,
            'osm_class_hierarchy': OSM_CLASS_HIERARCHY
        }
    }
    
    stats_output = os.path.join(output_dir, 'enrichment_stats.json')
    with open(stats_output, 'w') as f:
        json.dump(all_stats, f, indent=2, cls=NumpyEncoder)
    logger.info(f"  - Statistiques: {stats_output}")
    
    # ==========================================================================
    # RÉSUMÉ
    # ==========================================================================
    elapsed = datetime.now() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("ENRICHISSEMENT TERMINÉ")
    logger.info("=" * 60)
    logger.info(f"Durée totale: {elapsed}")
    logger.info(f"\nFichiers générés dans {output_dir}:")
    logger.info(f"  - nodes_enriched.csv ({len(nodes_enriched):,} nœuds)")
    logger.info(f"  - links_enriched.csv ({len(links_enriched):,} liens)")
    if travel_stats:
        logger.info(f"  - travel_times_enriched.csv ({travel_stats.get('total_records', 0):,} enregistrements)")
    logger.info(f"  - enrichment_stats.json")
    
    logger.info("\nFeatures générées (normalisées pour transfer learning):")
    logger.info("  Nœuds:")
    logger.info("    - centrality_percentile (0=centre, 1=périphérie)")
    logger.info("    - degree_in/out_normalized (Z-score)")
    logger.info("    - has_traffic_signal (binaire)")
    logger.info("  Liens:")
    logger.info("    - road_hierarchy (1-6, universel)")
    logger.info("    - length_percentile (rang relatif)")
    logger.info("    - avg_centrality (centralité du segment)")
    logger.info("    - estimated_capacity (capacité relative)")
    logger.info("  Temporel:")
    logger.info("    - hour_sin/cos, day_sin/cos (cyclique)")
    logger.info("    - is_weekend, is_rush_hour (binaire)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Enrichissement du graphe routier pour GNN - Phase 1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python enrich_graph.py --data-dir ./travel_times_2013 --output-dir ./enriched_data
  python enrich_graph.py -d ./data -o ./output --compute-neighbors
        """
    )
    parser.add_argument(
        '-d', '--data-dir',
        type=str,
        default='./travel_times_2013',
        help='Répertoire contenant nodes.csv, links.csv, travel_times_2013.csv'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='./enriched_data',
        help='Répertoire de sortie pour les fichiers enrichis'
    )
    parser.add_argument(
        '--compute-neighbors',
        action='store_true',
        help='Calculer les features de voisinage (coûteux en mémoire)'
    )
    
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.compute_neighbors)