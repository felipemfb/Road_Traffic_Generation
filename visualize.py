"""
Visualisation Interactive du Trafic Prédit
==========================================
Utilise le modèle Random Forest entraîné pour prédire et visualiser
le trafic sur n'importe quelle ville.

Usage:
    python visualize.py "Paris, France" --hour 8 --weekend
    python visualize.py "Lyon, France" --hour 18
    python visualize.py "Marseille, France" --hour 12 --weekend --output marseille_traffic.html
"""

import argparse
import os
import re
import warnings
from typing import Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import folium
from folium.plugins import Fullscreen
from branca.colormap import LinearColormap

warnings.filterwarnings('ignore')


# ============================================
# CONFIGURATION
# ============================================

MODEL_PATH = "models/traffic_rf_model.joblib"
METADATA_PATH = "models/model_metadata.joblib"

# Mapping des vitesses par défaut par type de route (km/h)
DEFAULT_SPEED_MAP = {
    "motorway": 110,
    "motorway_link": 80,
    "trunk": 100,
    "trunk_link": 70,
    "primary": 80,
    "primary_link": 60,
    "secondary": 60,
    "secondary_link": 50,
    "tertiary": 50,
    "tertiary_link": 40,
    "residential": 30,
    "living_street": 20,
    "service": 20,
    "unclassified": 30,
    "road": 30,
}

# Mapping place -> int (même que dans post_process.py)
PLACE_MAP = {
    "town": 0,
    "quarter": 1,
    "neighbourhood": 2, "cityblock": 2,
    "plot": 3,
    "village": 4,
    "hamlet": 5,
    "allotments": 6, "dwellings": 6, "farm": 6
}

# Landuse pertinents
RELEVANT_LANDUSE = [
    'commercial', 'retail', 'industrial', 'residential', 
    'construction', 'education', 'fairground', 'institutional'
]

RELEVANT_PLACES = [
    "town", "quarter", "neighbourhood", "cityblock", "plot", 
    "village", "hamlet", "allotments", "dwellings", "farm"
]


# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def get_city_epsg(city: str, country: str) -> int:
    """
    Retourne le code EPSG approprié pour une ville.
    Utilise UTM basé sur la longitude du centroïde.
    """
    try:
        # Récupérer le centroïde de la ville
        gdf = ox.geocode_to_gdf(f"{city}, {country}")
        centroid = gdf.geometry.centroid.iloc[0]
        lon = centroid.x
        lat = centroid.y
        
        # Calculer la zone UTM
        utm_zone = int((lon + 180) / 6) + 1
        
        # EPSG pour UTM Nord ou Sud
        if lat >= 0:
            epsg = 32600 + utm_zone  # UTM Nord
        else:
            epsg = 32700 + utm_zone  # UTM Sud
            
        return epsg
    except Exception:
        return 32618  # Par défaut (New York)


def parse_maxspeed_to_kmh(maxspeed) -> Optional[float]:
    """
    Convertit un tag OSM maxspeed en km/h.
    Retourne None si non exploitable.
    """
    if maxspeed is None or pd.isna(maxspeed):
        return None
    
    if isinstance(maxspeed, (int, float)):
        return float(maxspeed)
    
    if isinstance(maxspeed, list):
        maxspeed = maxspeed[0] if maxspeed else None
        if maxspeed is None:
            return None
    
    maxspeed = str(maxspeed).lower().strip()
    
    if maxspeed in {"signals", "walk", "none", "variable", "nan"}:
        return None
    
    # Prendre la première valeur si "50;70"
    maxspeed = maxspeed.split(";")[0].strip()
    
    # Extraire le nombre
    match = re.search(r"(\d+(\.\d+)?)", maxspeed)
    if not match:
        return None
    
    value = float(match.group(1))
    
    # Conversion selon unité
    if "mph" in maxspeed:
        return value * 1.60934
    elif "knot" in maxspeed:
        return value * 1.852
    else:
        return value  # km/h par défaut


def calculate_angle(geom) -> float:
    """
    Calcule l'angle d'un segment de route (différence entre début et fin).
    """
    if geom is None or geom.is_empty:
        return 0.0
    
    coords = list(geom.coords)
    if len(coords) < 2:
        return 0.0
    
    # Vecteur au début
    dx1 = coords[1][0] - coords[0][0]
    dy1 = coords[1][1] - coords[0][1]
    angle1 = np.degrees(np.arctan2(dy1, dx1))
    
    # Vecteur à la fin
    dx2 = coords[-1][0] - coords[-2][0]
    dy2 = coords[-1][1] - coords[-2][1]
    angle2 = np.degrees(np.arctan2(dy2, dx2))
    
    # Différence d'angle
    diff = abs(angle1 - angle2)
    return min(diff, 360 - diff)


# ============================================
# CHARGEMENT DU MODÈLE
# ============================================

def load_model() -> Tuple[object, dict]:
    """Charge le modèle et ses métadonnées."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modèle non trouvé: {MODEL_PATH}\n"
            "Exécutez d'abord random_forest.py pour entraîner et sauvegarder le modèle."
        )
    
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Métadonnées non trouvées: {METADATA_PATH}\n"
            "Exécutez d'abord random_forest.py pour sauvegarder les métadonnées."
        )
    
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    
    print(f"✓ Modèle chargé ({len(metadata['feature_columns'])} features)")
    
    return model, metadata


# ============================================
# RÉCUPÉRATION DES DONNÉES OSM
# ============================================

def fetch_city_data(city: str, country: str) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Récupère le graphe routier et les données contextuelles d'une ville.
    Retourne: edges_gdf, landuse_gdf, place_gdf
    """
    location = f"{city}, {country}"
    print(f"Récupération des données OSM pour {location}...")
    
    # Graphe routier
    print("  - Graphe routier...", end=" ", flush=True)
    G = ox.graph_from_place(location, network_type="drive", simplify=True)
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
    print(f"({len(edges_gdf)} segments)")
    
    # Landuse
    print("  - Zones landuse...", end=" ", flush=True)
    try:
        landuse_gdf = ox.features_from_place(location, tags={"landuse": True})
        landuse_gdf = landuse_gdf[landuse_gdf['landuse'].isin(RELEVANT_LANDUSE)]
        print(f"({len(landuse_gdf)} zones)")
    except Exception:
        landuse_gdf = gpd.GeoDataFrame(columns=['landuse', 'geometry'])
        print("(aucune)")
    
    # Place
    print("  - Zones place...", end=" ", flush=True)
    try:
        place_gdf = ox.features_from_place(location, tags={"place": True})
        place_gdf = place_gdf[place_gdf['place'].isin(RELEVANT_PLACES)]
        print(f"({len(place_gdf)} zones)")
    except Exception:
        place_gdf = gpd.GeoDataFrame(columns=['place', 'geometry'])
        print("(aucune)")
    
    return edges_gdf, landuse_gdf, place_gdf, nodes_gdf


# ============================================
# CONSTRUCTION DES FEATURES
# ============================================

def build_features(
    edges_gdf: gpd.GeoDataFrame,
    landuse_gdf: gpd.GeoDataFrame,
    place_gdf: gpd.GeoDataFrame,
    nodes_gdf: gpd.GeoDataFrame,
    metadata: dict,
    epsg: int,
    hour: int,
    is_weekend: int
) -> pd.DataFrame:
    """
    Construit les features pour chaque segment de route.
    """
    print("Construction des features...")
    
    # Copie et reset index
    df = edges_gdf.reset_index()
    
    # ---- road_type ----
    df['road_type'] = df['highway'].apply(
        lambda x: x[0] if isinstance(x, list) else x
    )
    
    # ---- max_speed ----
    df['max_speed'] = df['maxspeed'].apply(parse_maxspeed_to_kmh)
    # Imputation par type de route
    for road_type, default_speed in DEFAULT_SPEED_MAP.items():
        mask = (df['max_speed'].isna()) & (df['road_type'] == road_type)
        df.loc[mask, 'max_speed'] = default_speed
    df['max_speed'] = df['max_speed'].fillna(30)  # Défaut global
    
    # ---- lanes ----
    df['lanes'] = df['lanes'].apply(
        lambda x: int(x[0]) if isinstance(x, list) else (int(x) if pd.notna(x) else 1)
    )
    df['lanes'] = df['lanes'].fillna(1).astype(int)
    
    # ---- angle ----
    df['angle'] = df['geometry'].apply(calculate_angle)
    
    # ---- Projection pour calculs spatiaux ----
    df_projected = df.set_geometry('geometry').to_crs(epsg=epsg)
    
    # ---- centrality (distance au centre normalisée) ----
    # Centroïde de chaque segment
    df_projected['centroid'] = df_projected.geometry.centroid
    
    # Centre de la ville (centroïde global)
    all_points = df_projected['centroid'].union_all()
    city_center = all_points.centroid
    
    # Distance au centre
    df['dist_to_center'] = df_projected['centroid'].apply(
        lambda p: p.distance(city_center)
    )
    
    # Normalisation Z-score
    mu = df['dist_to_center'].mean()
    sigma = df['dist_to_center'].std()
    if sigma > 0:
        df['centrality'] = (df['dist_to_center'] - mu) / sigma
    else:
        df['centrality'] = 0
    
    # ---- num_begin_links et num_out_links ----
    # Compter les edges par nœud
    node_in_counts = df.groupby('v').size().to_dict()
    node_out_counts = df.groupby('u').size().to_dict()
    
    df['num_begin_links'] = df['u'].map(node_out_counts).fillna(1).astype(int)
    df['num_out_links'] = df['v'].map(node_in_counts).fillna(1).astype(int)
    
    # ---- landuse et place (jointure spatiale) ----
    # Centroïdes des segments pour la jointure
    edges_centroids = df_projected.copy()
    edges_centroids = edges_centroids.set_geometry('centroid')
    
    # Landuse
    if not landuse_gdf.empty:
        landuse_projected = landuse_gdf.to_crs(epsg=epsg)
        edges_with_landuse = gpd.sjoin_nearest(
            edges_centroids[['centroid']],
            landuse_projected[['landuse', 'geometry']],
            how='left',
            distance_col='dist_landuse'
        )
        # Garder seulement si proche (< 500m)
        edges_with_landuse.loc[edges_with_landuse['dist_landuse'] > 500, 'landuse'] = 'unknown'
        df['landuse'] = edges_with_landuse['landuse'].fillna('unknown').values
    else:
        df['landuse'] = 'unknown'
    
    # Place
    if not place_gdf.empty:
        place_projected = place_gdf.to_crs(epsg=epsg)
        edges_with_place = gpd.sjoin_nearest(
            edges_centroids[['centroid']],
            place_projected[['place', 'geometry']],
            how='left',
            distance_col='dist_place'
        )
        edges_with_place.loc[edges_with_place['dist_place'] > 1000, 'place'] = None
        df['place'] = edges_with_place['place'].map(PLACE_MAP).fillna(-1).astype(int).values
    else:
        df['place'] = -1
    
    # ---- landuse_way_dest et place_way_dest ----
    # Approximation: utiliser le landuse/place du nœud de destination
    df['landuse_way_dest'] = df['landuse']  # Simplification
    df['place_way_dest'] = df['place']
    
    # ---- Contexte temporel ----
    df['hour'] = hour
    df['is_weekend'] = is_weekend
    
    print(f"  ✓ {len(df)} segments avec features calculées")
    
    return df


def encode_features(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """
    Encode les features pour correspondre au format d'entraînement.
    """
    print("Encodage des features...")
    
    feature_columns = metadata['feature_columns']
    road_type_categories = metadata['road_type_categories']
    landuse_categories = metadata['landuse_categories']
    landuse_way_dest_categories = metadata['landuse_way_dest_categories']
    
    # Sélectionner les colonnes de base
    base_cols = ['place', 'place_way_dest', 'angle', 'max_speed', 'lanes', 
                 'centrality', 'num_begin_links', 'num_out_links', 'hour', 'is_weekend']
    
    X = df[base_cols].copy()
    
    # One-hot encoding manuel pour avoir les mêmes colonnes
    # road_type
    for cat in road_type_categories:
        col_name = f'road_type_{cat}'
        X[col_name] = (df['road_type'] == cat).astype(int)
    
    # landuse
    for cat in landuse_categories:
        col_name = f'landuse_{cat}'
        X[col_name] = (df['landuse'] == cat).astype(int)
    
    # landuse_way_dest
    for cat in landuse_way_dest_categories:
        col_name = f'landuse_way_dest_{cat}'
        X[col_name] = (df['landuse_way_dest'] == cat).astype(int)
    
    # S'assurer que toutes les colonnes du modèle sont présentes
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    
    # Réordonner selon l'ordre du modèle
    X = X[feature_columns]
    
    print(f"  ✓ {len(X.columns)} features encodées")
    
    return X


# ============================================
# PRÉDICTION
# ============================================

def predict_speeds(model, X: pd.DataFrame) -> np.ndarray:
    """Prédit les vitesses pour chaque segment."""
    print("Prédiction des vitesses...")
    
    speeds = model.predict(X)
    
    # Clipper les valeurs aberrantes
    speeds = np.clip(speeds, 5, 130)
    
    print(f"  ✓ Vitesse moyenne prédite: {speeds.mean():.1f} km/h")
    print(f"  ✓ Min: {speeds.min():.1f} km/h, Max: {speeds.max():.1f} km/h")
    
    return speeds


# ============================================
# VISUALISATION
# ============================================

def create_traffic_map(
    edges_gdf: gpd.GeoDataFrame,
    speeds: np.ndarray,
    city: str,
    hour: int,
    is_weekend: bool,
    output_path: str
) -> str:
    """
    Crée une carte Folium interactive avec les vitesses de trafic.
    """
    print("Création de la carte interactive...")
    
    # Convertir en WGS84 pour Folium
    edges_wgs84 = edges_gdf.to_crs(epsg=4326)
    
    # Centre de la carte
    bounds = edges_wgs84.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    # Créer la carte
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='cartodbpositron'
    )
    
    # Colormap: rouge (lent) -> jaune -> vert (rapide)
    colormap = LinearColormap(
        colors=['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850'],
        vmin=10,
        vmax=80,
        caption='Vitesse prédite (km/h)'
    )
    
    # Ajouter chaque segment
    for idx, (_, row) in enumerate(edges_wgs84.iterrows()):
        speed = speeds[idx]
        color = colormap(speed)
        
        # Épaisseur selon type de route
        highway = row.get('highway', 'residential')
        if isinstance(highway, list):
            highway = highway[0]
        
        weight_map = {
            'motorway': 5, 'trunk': 4, 'primary': 3.5,
            'secondary': 3, 'tertiary': 2.5, 'residential': 2,
            'motorway_link': 3, 'trunk_link': 2.5, 'primary_link': 2,
        }
        weight = weight_map.get(highway, 2)
        
        # Extraire les coordonnées
        if row.geometry.geom_type == 'LineString':
            coords = [(c[1], c[0]) for c in row.geometry.coords]
        else:
            continue
        
        # Info popup
        name = row.get('name', 'Route sans nom')
        if isinstance(name, list):
            name = name[0]
        
        popup_html = f"""
        <b>{name if pd.notna(name) else 'Route sans nom'}</b><br>
        Type: {highway}<br>
        <b>Vitesse prédite: {speed:.1f} km/h</b><br>
        Limite: {row.get('maxspeed', 'N/A')} km/h
        """
        
        folium.PolyLine(
            coords,
            color=color,
            weight=weight,
            opacity=0.8,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(m)
    
    # Ajouter la légende
    colormap.add_to(m)
    
    # Ajouter le titre
    day_type = "Weekend" if is_weekend else "Semaine"
    title_html = f'''
    <div style="position: fixed; 
                top: 10px; left: 50px; 
                background-color: white; 
                padding: 10px 20px; 
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.3);
                z-index: 9999;
                font-family: Arial, sans-serif;">
        <h3 style="margin: 0 0 5px 0;">🚗 Trafic Prédit - {city}</h3>
        <p style="margin: 0; color: #666;">
            {day_type}, {hour}h00
        </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Ajouter le bouton plein écran
    Fullscreen().add_to(m)
    
    # Sauvegarder
    m.save(output_path)
    print(f"  ✓ Carte sauvegardée: {output_path}")
    
    return output_path


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualisation du trafic prédit pour une ville"
    )
    parser.add_argument(
        "location",
        type=str,
        help="Ville et pays (ex: 'Paris, France')"
    )
    parser.add_argument(
        "--hour",
        type=int,
        default=8,
        help="Heure de la journée (0-23, défaut: 8)"
    )
    parser.add_argument(
        "--weekend",
        action="store_true",
        help="Simuler un jour de weekend"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Chemin du fichier HTML de sortie"
    )
    
    args = parser.parse_args()
    
    # Parser la location
    parts = args.location.split(",")
    if len(parts) < 2:
        print("Erreur: Spécifiez la ville et le pays (ex: 'Paris, France')")
        return
    
    city = parts[0].strip()
    country = ",".join(parts[1:]).strip()
    
    # Valider l'heure
    if not 0 <= args.hour <= 23:
        print("Erreur: L'heure doit être entre 0 et 23")
        return
    
    # Nom du fichier de sortie
    if args.output is None:
        city_slug = city.lower().replace(" ", "_")
        day_type = "weekend" if args.weekend else "semaine"
        args.output = f"traffic_{city_slug}_{day_type}_{args.hour}h.html"
    
    print("=" * 50)
    print(f"VISUALISATION DU TRAFIC - {city.upper()}")
    print(f"Jour: {'Weekend' if args.weekend else 'Semaine'}, Heure: {args.hour}h")
    print("=" * 50)
    
    # Charger le modèle
    model, metadata = load_model()
    
    # Récupérer les données OSM
    edges_gdf, landuse_gdf, place_gdf, nodes_gdf = fetch_city_data(city, country)
    
    # Déterminer l'EPSG
    epsg = get_city_epsg(city, country)
    print(f"Système de coordonnées: EPSG:{epsg}")
    
    # Construire les features
    df = build_features(
        edges_gdf, landuse_gdf, place_gdf, nodes_gdf,
        metadata, epsg, args.hour, int(args.weekend)
    )
    
    # Encoder
    X = encode_features(df, metadata)
    
    # Prédire
    speeds = predict_speeds(model, X)
    
    # Visualiser
    output_path = create_traffic_map(
        edges_gdf, speeds, city, args.hour, args.weekend, args.output
    )
    
    print("\n" + "=" * 50)
    print(f"✓ Terminé! Ouvrez {output_path} dans votre navigateur")
    print("=" * 50)


if __name__ == "__main__":
    main()
