import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
import joblib
from shapely.geometry import LineString


CITY = "Marseille, France"
MODEL_PATH = "models/random_forest_ny.joblib"
OUTPUT_PATH = "data/processed/marseille_predictions.csv"

# Télécharger le graphe OSM
print("Downloading Marseille road network...")
G = ox.graph_from_place(CITY, network_type="drive")

nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
edges = edges.reset_index()

edges = edges[["u", "v", "length", "highway", "lanes", "maxspeed", "geometry"]]

# Feature engineering
edges["street_length"] = edges["length"]

# Conversion de maxspeed en km/h
def parse_maxspeed(x):
    if isinstance(x, list):
        x = x[0]
    if not isinstance(x, str):
        return np.nan
    x = x.lower()
    import re
    m = re.search(r"\d+", x)
    if not m:
        return np.nan
    v = float(m.group())
    if "mph" in x:
        v *= 1.609
    elif "knot" in x:
        v *= 1.852
    return v

edges["max_speed"] = edges["maxspeed"].apply(parse_maxspeed)

edges["lanes"] = pd.to_numeric(edges["lanes"], errors="coerce")

def compute_angle(geom):
    if geom is None or geom.is_empty:
        return 0
    x1, y1 = geom.coords[0]
    x2, y2 = geom.coords[-1]
    return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 360

edges["angle"] = edges["geometry"].apply(compute_angle)

G_simple = G.to_undirected()
degree = dict(G_simple.degree())

edges["num_begin_links"] = edges["u"].map(degree)
edges["num_out_links"] = edges["v"].map(degree)

cent = nx.closeness_centrality(G_simple)
edges["centrality"] = edges.apply(lambda row: (cent.get(row["u"], 0) + cent.get(row["v"], 0)) / 2, axis=1)

# On ajoute des features temporelles factices (le modèle en a besoin)
edges["hour"] = 12
edges["is_weekend"] = 0

edges["place"] = 2
edges["place_way_dest"] = 2

for lu in ["commercial","construction","education","industrial","residential","retail"]:
    edges[f"landuse_{lu}"] = 0
    edges[f"landuse_way_dest_{lu}"] = 0

# One-hot encoding des features catégorielles
road_types = [
    "living_street","motorway","motorway_link","primary","primary_link",
    "residential","road","secondary","secondary_link","tertiary",
    "tertiary_link","trunk","trunk_link","unclassified"
]

for rt in road_types:
    edges[f"road_type_{rt}"] = (edges["highway"] == rt).astype(int)

# Aligner les features avec le modèle de NYC
model_features = pd.read_csv("data/processed/training.csv").drop("speed", axis=1).columns
X_marseille = edges.reindex(columns=model_features).fillna(0)

# Load le model entraîné
print("Loading trained model...")
model = joblib.load(MODEL_PATH)

# Prédire les vitesses à Marseille
print("Predicting Marseille speeds...")
edges["predicted_speed"] = model.predict(X_marseille)

# Sauvegarder les prédictions
edges[["u", "v", "geometry", "predicted_speed"]].to_file(
    OUTPUT_PATH.replace(".csv", ".geojson"),
    driver="GeoJSON"
)

edges[["u", "v", "predicted_speed"]].to_csv(OUTPUT_PATH, index=False)

print("Saved Marseille predictions:")
print(" →", OUTPUT_PATH)
print(" →", OUTPUT_PATH.replace(".csv", ".geojson"))
