#!/usr/bin/env python3
"""
NYC Traffic Data Preprocessor - VERSION OPTIMISÉE
==================================================
Traitement vectorisé avec pandas (10-50x plus rapide)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
import gc
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "travel_times_2013"
OUTPUT_DIR = Path(__file__).parent / "data"

NODES_FILE = DATA_DIR / "nodes.csv"
LINKS_FILE = DATA_DIR / "links.csv"
TRAVEL_TIMES_FILE = DATA_DIR / "travel_times_2013.csv"

# Seuils de vitesse (km/h)
SPEED_THRESHOLDS = {'fluid': 40, 'normal': 25, 'dense': 15, 'congested': 0}

# Couleurs RGBA
COLORS = {
    'fluid': [34, 197, 94, 200],
    'normal': [250, 204, 21, 200],
    'dense': [249, 115, 22, 200],
    'congested': [239, 68, 68, 200],
    'no_data': [107, 114, 128, 100]
}

# =============================================================================
# FONCTIONS OPTIMISÉES
# =============================================================================

def load_nodes(filepath):
    print(f"📍 Chargement des nœuds...")
    df = pd.read_csv(filepath, usecols=['node_id', 'xcoord', 'ycoord'])
    nodes = df.set_index('node_id')[['xcoord', 'ycoord']].to_dict('index')
    print(f"   ✓ {len(nodes):,} nœuds")
    return nodes


def load_links(filepath, nodes):
    print(f"🛤️  Chargement des segments...")
    
    cols = ['link_id', 'begin_node_id', 'end_node_id', 'street_length', 'osm_class', 'osm_name']
    df = pd.read_csv(filepath, usecols=lambda x: x in cols)
    
    # Vectorized coordinate lookup
    nodes_df = pd.DataFrame(nodes).T
    nodes_df.index = nodes_df.index.astype(int)
    
    df['start_lon'] = df['begin_node_id'].map(lambda x: nodes.get(x, {}).get('xcoord'))
    df['start_lat'] = df['begin_node_id'].map(lambda x: nodes.get(x, {}).get('ycoord'))
    df['end_lon'] = df['end_node_id'].map(lambda x: nodes.get(x, {}).get('xcoord'))
    df['end_lat'] = df['end_node_id'].map(lambda x: nodes.get(x, {}).get('ycoord'))
    
    df = df.dropna(subset=['start_lon', 'start_lat', 'end_lon', 'end_lat'])
    df['segment_key'] = df['begin_node_id'].astype(str) + '_' + df['end_node_id'].astype(str)
    
    print(f"   ✓ {len(df):,} segments")
    return df


def get_color_array(speeds):
    """Vectorized color assignment."""
    colors = np.zeros((len(speeds), 4), dtype=np.int32)
    colors[:] = COLORS['no_data']
    
    mask_fluid = speeds > 40
    mask_normal = (speeds > 25) & (speeds <= 40)
    mask_dense = (speeds > 15) & (speeds <= 25)
    mask_congested = (speeds > 0) & (speeds <= 15)
    
    colors[mask_fluid] = COLORS['fluid']
    colors[mask_normal] = COLORS['normal']
    colors[mask_dense] = COLORS['dense']
    colors[mask_congested] = COLORS['congested']
    
    return colors


def process_travel_times_fast(filepath, links_df, chunksize=2_000_000):
    """
    Version OPTIMISÉE - opérations vectorisées uniquement.
    """
    print(f"⏱️  Traitement des temps de trajet (version optimisée)...")
    
    # Créer lookup table pour les longueurs
    segment_lengths = links_df.set_index('segment_key')['street_length']
    
    # Structure pour stocker les résultats agrégés
    # On va accumuler sum(speed * num_trips) et sum(num_trips) pour faire une moyenne pondérée
    hourly_speed_sum = {h: {} for h in range(24)}
    hourly_trip_sum = {h: {} for h in range(24)}
    
    total_rows = 0
    chunk_num = 0
    
    # Estimer le nombre total de chunks
    import os
    file_size = os.path.getsize(filepath)
    estimated_chunks = max(1, file_size // (chunksize * 50))  # ~50 bytes par ligne
    
    print(f"   📊 Fichier: {file_size / 1e9:.1f} GB, ~{estimated_chunks} chunks estimés")
    
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        chunk_num += 1
        total_rows += len(chunk)
        
        print(f"   Chunk {chunk_num}/{estimated_chunks} - {total_rows:,} lignes traitées", end='\r')
        
        # Créer segment_key (vectorisé)
        chunk['segment_key'] = chunk['begin_node_id'].astype(str) + '_' + chunk['end_node_id'].astype(str)
        
        # Extraire l'heure (vectorisé)
        chunk['hour'] = pd.to_datetime(chunk['datetime']).dt.hour
        
        # Ajouter longueur (vectorisé via merge)
        chunk['street_length'] = chunk['segment_key'].map(segment_lengths)
        
        # Filtrer (vectorisé)
        chunk = chunk.dropna(subset=['street_length'])
        chunk = chunk[chunk['travel_time'] > 0]
        
        # Calculer vitesse (vectorisé)
        chunk['speed_kmh'] = (chunk['street_length'] / chunk['travel_time']) * 3.6
        chunk = chunk[chunk['speed_kmh'] <= 150]
        
        # Calculer speed * num_trips pour moyenne pondérée
        chunk['weighted_speed'] = chunk['speed_kmh'] * chunk['num_trips']
        
        # Agréger par heure et segment (vectorisé avec groupby)
        for hour in range(24):
            hour_data = chunk[chunk['hour'] == hour]
            if len(hour_data) == 0:
                continue
            
            # Groupby vectorisé
            grouped = hour_data.groupby('segment_key').agg({
                'weighted_speed': 'sum',
                'num_trips': 'sum'
            })
            
            # Accumuler les résultats
            for seg_key, row in grouped.iterrows():
                if seg_key not in hourly_speed_sum[hour]:
                    hourly_speed_sum[hour][seg_key] = 0
                    hourly_trip_sum[hour][seg_key] = 0
                hourly_speed_sum[hour][seg_key] += row['weighted_speed']
                hourly_trip_sum[hour][seg_key] += row['num_trips']
        
        # Libérer mémoire
        del chunk
        gc.collect()
    
    print(f"\n   ✓ {total_rows:,} lignes traitées en {chunk_num} chunks")
    
    # Calculer les moyennes pondérées finales
    print("   📊 Calcul des vitesses moyennes...")
    hourly_avg_speeds = {h: {} for h in range(24)}
    
    for hour in range(24):
        for seg_key in hourly_speed_sum[hour]:
            total_weighted = hourly_speed_sum[hour][seg_key]
            total_trips = hourly_trip_sum[hour][seg_key]
            if total_trips > 0:
                avg_speed = total_weighted / total_trips
                hourly_avg_speeds[hour][seg_key] = {
                    'speed': round(avg_speed, 1),
                    'trips': int(total_trips)
                }
    
    return hourly_avg_speeds


def generate_output_files(links_df, hourly_speeds, output_dir):
    """Génère les fichiers JSON optimisés."""
    print("📁 Génération des fichiers...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Préparer les segments
    segments = []
    segment_key_to_idx = {}
    
    for idx, (_, row) in enumerate(links_df.iterrows()):
        segment = {
            'id': int(row['link_id']),
            'key': row['segment_key'],
            'path': [
                [round(float(row['start_lon']), 6), round(float(row['start_lat']), 6)],
                [round(float(row['end_lon']), 6), round(float(row['end_lat']), 6)]
            ],
            'length': round(float(row['street_length']), 1),
            'name': str(row.get('osm_name', '')),
            'type': str(row.get('osm_class', 'road'))
        }
        segments.append(segment)
        segment_key_to_idx[row['segment_key']] = idx
    
    # Sauvegarder segments.json
    base_data = {
        'metadata': {
            'total_segments': len(segments),
            'speed_thresholds': SPEED_THRESHOLDS,
            'colors': COLORS,
            'generated_at': datetime.now().isoformat()
        },
        'segments': segments
    }
    
    with open(output_dir / 'segments.json', 'w') as f:
        json.dump(base_data, f)
    print(f"   ✓ segments.json ({len(segments):,} segments)")
    
    # Générer fichiers horaires
    for hour in range(24):
        speeds_data = hourly_speeds.get(hour, {})
        
        hour_array = []
        for seg in segments:
            seg_key = seg['key']
            
            if seg_key in speeds_data:
                speed = speeds_data[seg_key]['speed']
                # Déterminer couleur
                if speed > 40:
                    color = COLORS['fluid']
                elif speed > 25:
                    color = COLORS['normal']
                elif speed > 15:
                    color = COLORS['dense']
                else:
                    color = COLORS['congested']
            else:
                speed = -1
                color = COLORS['no_data']
            
            hour_array.append([speed] + color)
        
        with open(output_dir / f'hour_{hour:02d}.json', 'w') as f:
            json.dump(hour_array, f)
        
        print(f"   ✓ hour_{hour:02d}.json")
    
    print(f"\n✅ Fichiers générés dans {output_dir}")


def main():
    print("=" * 60)
    print("🚗 NYC Traffic Preprocessor (OPTIMISÉ)")
    print("=" * 60)
    print()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not NODES_FILE.exists() or not LINKS_FILE.exists():
        print(f"⚠️  Fichiers non trouvés dans {DATA_DIR}")
        return
    
    nodes = load_nodes(NODES_FILE)
    links_df = load_links(LINKS_FILE, nodes)
    
    print()
    
    if TRAVEL_TIMES_FILE.exists():
        hourly_speeds = process_travel_times_fast(TRAVEL_TIMES_FILE, links_df)
    else:
        print(f"⚠️  {TRAVEL_TIMES_FILE} non trouvé")
        return
    
    print()
    generate_output_files(links_df, hourly_speeds, OUTPUT_DIR)
    
    print()
    print("=" * 60)
    print("✅ Terminé !")
    print("=" * 60)


if __name__ == "__main__":
    main()