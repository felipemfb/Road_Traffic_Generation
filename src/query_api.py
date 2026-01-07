"""
Service communiquant avec l'api d'osm
"""
import osmnx as ox
import pandas as pd

def get_geodata(city: str, country: str):
    try:
        gdf_landuse = ox.features_from_place(f"{city}, {country}", tags={"landuse": True})
        gdf_place = ox.features_from_place(f"{city}, {country}", tags={"place": True})
    except Exception as e:
        print("failed to query gdf with osmnx")

    return gdf_landuse, gdf_place

def get_ways(city):
    """
    Récupère toutes les ways routières pour une ville avec osmnx,
    et retourne un DataFrame avec seulement: osm_way_id, maxspeed, lanes.
    """
    # Récupérer le graphe routier "drivable"
    G = ox.graph_from_place(city, network_type="drive", simplify=True)
    
    # Convertir les arêtes du graphe en GeoDataFrame
    edges = ox.graph_to_gdfs(G, nodes=False)
    
    # Garder seulement les colonnes nécessaires
    waysDf = edges.reset_index()[["osmid", "maxspeed", "lanes"]]
    
    # Renommer la colonne osmid pour correspondre à ton pipeline
    waysDf = waysDf.rename(columns={"osmid": "osm_way_id"})

    waysDf = waysDf.explode("osm_way_id")
    waysDf["osm_way_id"] = waysDf["osm_way_id"].astype(int)

    waysDf['lanes'] = waysDf['lanes'].apply(lambda x: x[0] if isinstance(x, list) else x)
    waysDf['lanes'] = waysDf['lanes'].astype(str).str.split(';').str[0].astype(float).fillna(1).astype(int)

    waysDf['maxspeed'] = waysDf['maxspeed'].apply(lambda x: x[0] if isinstance(x, list) else x)
    waysDf['maxspeed'] = waysDf['maxspeed'].astype(str).str.split(';').str[0]

    waysDf = waysDf.groupby('osm_way_id').agg({
        'maxspeed': 'first',
        'lanes': 'first'
    }).reset_index()

    return waysDf
