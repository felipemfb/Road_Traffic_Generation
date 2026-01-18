import osmnx as ox
import pandas as pd
import folium
import branca.colormap as cm
import numpy as np

CITY = "Marseille, France"
PRED_PATH = "data/processed/marseille_predictions.csv"
OUTPUT_HTML = "marseille_speed_map.html"

# Télécharger les prédictions

pred = pd.read_csv(PRED_PATH)
pred["key"] = list(zip(pred.u, pred.v))
pred_dict = dict(zip(pred.key, pred.predicted_speed))

# Télécharger le graphe de Marseille

print("Downloading Marseille network...")
G = ox.graph_from_place(CITY, network_type="drive")

# Ajouter les prédictions au graphe

for u, v, k, data in G.edges(keys=True, data=True):
    data["predicted_speed"] = pred_dict.get((u, v), np.nan)

# Faire la conversion en GeoDataFrame

edges = ox.graph_to_gdfs(G, nodes=False)
edges = edges.dropna(subset=["predicted_speed"])

# Colormap (vert → jaune → rouge)

vmin = edges["predicted_speed"].quantile(0.05)
vmax = edges["predicted_speed"].quantile(0.95)

colormap = cm.LinearColormap(
    colors=["#d50000", "#ffd600", "#00c853"],
    vmin=vmin,
    vmax=vmax,
    text_color = 'white'
)

# Créer une carte noire

center = edges.geometry.unary_union.centroid
m = folium.Map(
    location=[center.y, center.x],
    zoom_start=12,
    tiles="CartoDB dark_matter"
)

# Ajouter les rues 

for _, row in edges.iterrows():
    geom = row.geometry
    if geom is None:
        continue

    speed = row.predicted_speed
    color = colormap(speed)

    name = row.get("name", "Unnamed road")

    popup = f"""
    <b>{name}</b><br>
    Predicted speed: {speed:.1f} km/h
    """

    gj = folium.GeoJson(
        geom,
        style_function=lambda x, color=color: {
            "color": color,
            "weight": 3,
            "opacity": 0.9
        },
        tooltip=popup
    )
    gj.add_to(m)

# Ajouter la légende

colormap.caption = "Predicted speed (km/h)"
colormap.add_to(m)

# Sauvegarder

m.save(OUTPUT_HTML)

print("Map created:")
print("Open in browser:", OUTPUT_HTML)
