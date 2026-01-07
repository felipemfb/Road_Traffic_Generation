"""
-----------------------------------------------
2: Enrichissement depuis open_street_map
-----------------------------------------------
Objectif: récupérer nos features depuis l'api d'osm
Features à récupérer:
    - landuse
    - landuse_way_dest
    - place
    - place_way_dest
    - max_speed
    - num_lanes
    - centrality
-------- ways --------
    max_speed
    lanes
--------- nodes ---------
    landuse
    place
    centrality
Imputer si absent de ways:
    landuse_way_dest
    place_way_dest
"""

import pandas as pd
import geopandas as gpd
import query_api

import matplotlib.pyplot as plt

def convert_to_gdf(nodesDf, EPSG):
    nodesGdf = gpd.GeoDataFrame(
        nodesDf,
        geometry=gpd.points_from_xy(nodesDf["xcoord"], nodesDf["ycoord"]),
        crs="EPSG:4326" # osm / lat-lon
    )

    nodesGdf = nodesGdf.to_crs(epsg=EPSG)

    return nodesGdf

def enrich_nodes_with_tags(nodesGdf, gdf_landuse, gdf_place, EPSG):
    gdf_landuse = gdf_landuse.to_crs(epsg=EPSG)
    gdf_place = gdf_place.to_crs(epsg=EPSG)

    # landuse
    relevant_landuse = ['commercial', 'retail', 'industrial', 'residential', 'construction', 'education', 'fairground', 'institutional'] # type de landuse importants
    gdf_landuse_filtered = gdf_landuse[gdf_landuse['landuse'].isin(relevant_landuse)]

    nodesGdf = gpd.sjoin_nearest(
        nodesGdf,
        gdf_landuse_filtered[["landuse", "geometry"]],
        how="left",
        distance_col="dist_to_landuse"
    )

    # place
    relevant_places = ["town", "quarter", "neighbourhood", "cityblock", "plot", "village", "hamlet", "allotments", "dwellings", "farm"]
    gdf_place_filtered = gdf_place[gdf_place['place'].isin(relevant_places)]

    nodesGdf = gpd.sjoin_nearest(
        nodesGdf,
        gdf_place_filtered[["place", "geometry"]],
        how="left",
        distance_col="dist_to_place"
    )

    nodesGdf = nodesGdf.drop_duplicates(subset="node_id", keep="first")

    return nodesGdf

def enrich_nodes_with_dist_to_center(nodesGdf):
    """
    Appeler apres enrich_nodes_with_tags
    """
    # centroid du graphe
    center = nodesGdf.geometry.union_all().centroid
    center_x, center_y = center.x, center.y

    # distance euclidienne
    nodesGdf['dist_to_center'] = nodesGdf.geometry.apply(
        lambda geom: ((geom.x - center_x)**2 + (geom.y - center_y)**2) ** 0.5
    )

    return nodesGdf

def enrich_ways_max_speed_and_lanes(waysDf, ways_tagsDf):
    waysDf = waysDf.merge(
        ways_tagsDf,
        on="osm_way_id",
        how="left"
    )

    # Gestion des valeurs nulles
    waysDf['maxspeed'] = waysDf['maxspeed'].fillna('unknown')
    waysDf['lanes'] = waysDf['lanes'].fillna(1).astype(int)

    return waysDf


# gdf_landuse, gdf_place = query_api.get_geodata("New York", "USA")

# nodesDf = pd.read_csv("data/raw/nodes.csv")

# nodesGdf = enrich_nodes_with_tags(nodesDf, gdf_landuse, gdf_place, 32618)

# nodesGdf = enrich_nodes_with_dist_to_center(nodesGdf)

# print(nodesGdf["dist_to_center"].describe())

# waysDf = pd.read_csv("data/raw/nodes.csv")