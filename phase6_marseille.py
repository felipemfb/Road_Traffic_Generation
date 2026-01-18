import pandas as pd
import numpy as np
import os
import pickle
import joblib
from scipy.spatial import KDTree

try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False
    print("ERREUR: osmnx non installé.")
    exit(1)

# CONFIGURATION

OUTPUT_DIR = "marseille_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_FILE = "models/model_best_nyc.pkl"
FEATURE_LIST_FILE = "models/feature_list.txt"

MARSEILLE_GRAPH_CACHE = os.path.join(OUTPUT_DIR, "marseille_graph.pkl")
MARSEILLE_POI_CACHE = os.path.join(OUTPUT_DIR, "marseille_pois.pkl")

OSM_CLASS_ORDER = {
    'motorway': 1, 'motorway_link': 2,
    'trunk': 3, 'trunk_link': 4,
    'primary': 5, 'primary_link': 6,
    'secondary': 7, 'secondary_link': 8,
    'tertiary': 9, 'tertiary_link': 10,
    'unclassified': 11, 'residential': 12,
    'living_street': 13, 'service': 14
}


print("=" * 60)
print("PHASE 6 : APPLICATION À MARSEILLE (CORRIGÉ)")
print("=" * 60)

# Charger la liste des features attendues

print("\n--- Chargement de la liste des features ---\n")

if os.path.exists(FEATURE_LIST_FILE):
    with open(FEATURE_LIST_FILE, 'r') as f:
        feature_list = [line.strip() for line in f if line.strip()]
    print(f"  Features attendues par le modèle : {len(feature_list)}")
    print(f"  {feature_list}")
else:
    print("  ERREUR: feature_list.txt non trouvé!")
    exit(1)

# Téléchargement du réseau routier

print("\n--- Téléchargement du réseau routier ---\n")

if os.path.exists(MARSEILLE_GRAPH_CACHE):
    print("Chargement du graphe depuis le cache...")
    with open(MARSEILLE_GRAPH_CACHE, 'rb') as f:
        G = pickle.load(f)
else:
    print("Téléchargement du réseau routier de Marseille...")
    G = ox.graph_from_place("Marseille, France", network_type="drive")
    with open(MARSEILLE_GRAPH_CACHE, 'wb') as f:
        pickle.dump(G, f)

nodes, edges = ox.graph_to_gdfs(G)
print(f"  Segments : {len(edges)}")
print(f"  Noeuds : {len(nodes)}")


# Téléchargement des POI

print("\n--- Téléchargement des POI ---\n")

def load_or_download_marseille_pois():
    if os.path.exists(MARSEILLE_POI_CACHE):
        print("Chargement des POI depuis le cache...")
        with open(MARSEILLE_POI_CACHE, 'rb') as f:
            return pickle.load(f)
    
    print("Téléchargement des POI de Marseille...")
    pois = {}
    
    try:
        pois['attractors'] = ox.features_from_place("Marseille, France", 
            tags={"amenity": ["school", "college", "university"]})
        print(f"  Attracteurs: {len(pois['attractors'])}")
    except: pois['attractors'] = None
    
    try:
        pois['transport'] = ox.features_from_place("Marseille, France",
            tags={"railway": ["station", "subway_entrance", "tram_stop"]})
        print(f"  Transport: {len(pois['transport'])}")
    except: pois['transport'] = None
    
    try:
        pois['parks'] = ox.features_from_place("Marseille, France",
            tags={"leisure": "park"})
        print(f"  Parcs: {len(pois['parks'])}")
    except: pois['parks'] = None
    
    with open(MARSEILLE_POI_CACHE, 'wb') as f:
        pickle.dump(pois, f)
    return pois

pois = load_or_download_marseille_pois()

# Préparation des segments

print("\n--- Préparation des segments ---\n")

edges_df = edges.reset_index()
segments = pd.DataFrame()

segments['u'] = edges_df['u']
segments['v'] = edges_df['v']

# street_length
if 'length' in edges_df.columns:
    segments['street_length'] = edges_df['length']
else:
    segments['street_length'] = edges_df.geometry.length * 111000

# osm_class
if 'highway' in edges_df.columns:
    def get_highway_class(x):
        return x[0] if isinstance(x, list) else x
    segments['osm_class'] = edges_df['highway'].apply(get_highway_class)
else:
    segments['osm_class'] = 'unclassified'

# Coordonnées
segments['startX'] = edges_df.geometry.apply(lambda g: g.coords[0][0])
segments['startY'] = edges_df.geometry.apply(lambda g: g.coords[0][1])
segments['endX'] = edges_df.geometry.apply(lambda g: g.coords[-1][0])
segments['endY'] = edges_df.geometry.apply(lambda g: g.coords[-1][1])

print(f"  Segments préparés : {len(segments)}")


# Features géométriques

print("\n--- Features géométriques ---\n")

# log_street_length
segments['log_street_length'] = np.log1p(segments['street_length'])

# Calculer les angles de début et fin
dx = segments['endX'] - segments['startX']
dy = segments['endY'] - segments['startY']
segments['begin_angle'] = np.degrees(np.arctan2(dy, dx))
segments['end_angle'] = segments['begin_angle'] + 180  # Segment droit par défaut

# Curvature = écart par rapport à segment droit (0 = droit)
angle_diff = segments['end_angle'] - segments['begin_angle']
segments['curvature'] = np.abs(180 - np.abs(angle_diff))
print(f"  curvature: min={segments['curvature'].min():.2f}, max={segments['curvature'].max():.2f}, mean={segments['curvature'].mean():.2f}")

# osm_class_encoded
segments['osm_class_encoded'] = segments['osm_class'].map(OSM_CLASS_ORDER).fillna(15)

# Features topologiques

print("\n--- Features topologiques ---\n")

in_degree = dict(G.in_degree())
out_degree = dict(G.out_degree())

segments['begin_num_in_links'] = segments['u'].map(in_degree).fillna(1)
segments['begin_num_out_links'] = segments['u'].map(out_degree).fillna(1)
segments['end_num_in_links'] = segments['v'].map(in_degree).fillna(1)
segments['end_num_out_links'] = segments['v'].map(out_degree).fillna(1)

segments['begin_node_degree'] = segments['begin_num_in_links'] + segments['begin_num_out_links']
segments['end_node_degree'] = segments['end_num_in_links'] + segments['end_num_out_links']
segments['avg_node_degree'] = (segments['begin_node_degree'] + segments['end_node_degree']) / 2

print("  Degrés des noeuds calculés")


# Features de position

print("\n--- Features de position ---\n")

segments['centroid_x'] = (segments['startX'] + segments['endX']) / 2
segments['centroid_y'] = (segments['startY'] + segments['endY']) / 2

city_center_x = segments['centroid_x'].mean()
city_center_y = segments['centroid_y'].mean()
print(f"  Centre de Marseille : ({city_center_x:.4f}, {city_center_y:.4f})")

segments['dist_to_center'] = np.sqrt(
    (segments['centroid_x'] - city_center_x)**2 +
    (segments['centroid_y'] - city_center_y)**2
)

max_dist = segments['dist_to_center'].max()
segments['dist_to_center_norm'] = segments['dist_to_center'] / max_dist

# Features POI

print("\n--- Features POI ---\n")

def extract_poi_coords(gdf):
    if gdf is None or len(gdf) == 0:
        return None
    coords = []
    for geom in gdf.geometry:
        try:
            if geom.geom_type == 'Point':
                coords.append((geom.y, geom.x))
            elif geom.geom_type in ['Polygon', 'MultiPolygon']:
                coords.append((geom.centroid.y, geom.centroid.x))
        except: continue
    return np.array(coords) if coords else None


def calc_poi_features(segment_coords, poi_coords):
    if poi_coords is None or len(poi_coords) == 0:
        return pd.Series([np.nan] * len(segment_coords)), pd.Series([0] * len(segment_coords))
    tree = KDTree(poi_coords)
    distances, _ = tree.query(segment_coords)
    counts = tree.query_ball_point(segment_coords, r=0.005, return_length=True)
    return pd.Series(distances), pd.Series(counts)


segment_coords = np.column_stack([segments['centroid_y'].values, segments['centroid_x'].values])

for poi_type, prefix in [('attractors', 'attractor'), ('transport', 'transport'), ('parks', 'park')]:
    coords = extract_poi_coords(pois.get(poi_type))
    segments[f'dist_{prefix}'], segments[f'count_{prefix}s_500m'] = calc_poi_features(segment_coords, coords)
    print(f"  dist_{prefix}, count_{prefix}s_500m : calculés")


# Densité locale

print("\n--- Densité locale ---\n")

segment_tree = KDTree(segment_coords)
segments['road_density_500m'] = segment_tree.query_ball_point(segment_coords, r=0.005, return_length=True)

# Combinaisons temporelles

print("\n--- Génération des combinaisons temporelles ---\n")

temporal_combinations = []
for day in range(7):
    for hour in range(24):
        is_weekend = 1 if day >= 5 else 0
        if 7 <= hour <= 9:
            period = 1
        elif 17 <= hour <= 19:
            period = 2
        elif hour >= 22 or hour <= 6:
            period = 3
        else:
            period = 0
        temporal_combinations.append({'period': period, 'is_weekend': is_weekend})

temporal_df = pd.DataFrame(temporal_combinations).drop_duplicates()
print(f"  Combinaisons uniques : {len(temporal_df)}")

# Produit cartésien simplifié (seulement period × is_weekend)
n_segments = len(segments)
n_times = len(temporal_df)

dataset = pd.concat([segments] * n_times, ignore_index=True)
temporal_tiled = pd.DataFrame()
for col in temporal_df.columns:
    temporal_tiled[col] = np.tile(temporal_df[col].values, n_segments)

# Réorganiser correctement
temporal_repeated = pd.concat([temporal_df] * n_segments, ignore_index=True)
dataset = pd.concat([segments.loc[segments.index.repeat(n_times)].reset_index(drop=True), 
                     temporal_repeated.reset_index(drop=True)], axis=1)

print(f"  Dataset final : {len(dataset):,} lignes")

# Prédiction

print("\n--- Chargement du modèle et prédiction ---\n")

model = joblib.load(MODEL_FILE)
print(f"  Modèle chargé : {MODEL_FILE}")

# Vérifier les features
available_features = [f for f in feature_list if f in dataset.columns]
missing_features = [f for f in feature_list if f not in dataset.columns]

if missing_features:
    print(f"  WARNING - Features manquantes : {missing_features}")
    for f in missing_features:
        dataset[f] = np.nan

print(f"  Features utilisées : {len(available_features)}/{len(feature_list)}")

X_marseille = dataset[feature_list].copy()
X_marseille = X_marseille.fillna(X_marseille.median())
X_marseille = X_marseille.replace([np.inf, -np.inf], 0)

print("  Prédiction en cours...")
vitesse_predite = model.predict(X_marseille)
vitesse_predite = np.clip(vitesse_predite, 1, 120)

dataset['speed_kmh_predicted'] = vitesse_predite
print(f"  Prédictions terminées : {len(vitesse_predite):,} valeurs")


# Statistiques

print("\n--- Statistiques des prédictions ---\n")

print(f"Distribution de la vitesse prédite (km/h) :")
print(f"  Min    : {vitesse_predite.min():.2f}")
print(f"  Max    : {vitesse_predite.max():.2f}")
print(f"  Moyenne: {vitesse_predite.mean():.2f}")
print(f"  Médiane: {np.median(vitesse_predite):.2f}")
print(f"  Std    : {vitesse_predite.std():.2f}")

period_names = {0: 'Heures creuses', 1: 'Rush matin', 2: 'Rush soir', 3: 'Nuit'}
print("\nVitesse moyenne par période :")
for period_id, period_name in period_names.items():
    mask = dataset['period'] == period_id
    if mask.sum() > 0:
        mean_speed = dataset.loc[mask, 'speed_kmh_predicted'].mean()
        print(f"  {period_name}: {mean_speed:.2f} km/h")

print("\nVitesse moyenne semaine/weekend :")
print(f"  Semaine : {dataset.loc[dataset['is_weekend']==0, 'speed_kmh_predicted'].mean():.2f} km/h")
print(f"  Weekend : {dataset.loc[dataset['is_weekend']==1, 'speed_kmh_predicted'].mean():.2f} km/h")

# Sauvegarde

print("\n--- Sauvegarde ---\n")

cols_to_save = ['u', 'v', 'osm_class', 'street_length', 'period', 'is_weekend', 'speed_kmh_predicted']
dataset[cols_to_save].to_csv(os.path.join(OUTPUT_DIR, "marseille_predictions.csv"), index=False)

summary = dataset.groupby(['u', 'v', 'osm_class']).agg({
    'street_length': 'first',
    'speed_kmh_predicted': ['mean', 'min', 'max', 'std']
}).reset_index()
summary.columns = ['u', 'v', 'osm_class', 'street_length', 'speed_mean', 'speed_min', 'speed_max', 'speed_std']
summary.to_csv(os.path.join(OUTPUT_DIR, "marseille_summary.csv"), index=False)

print(f"  Fichiers sauvegardés dans {OUTPUT_DIR}/")

print("\n" + "=" * 60)
print("PHASE 6 TERMINÉE (CORRIGÉE)")
print("=" * 60)
