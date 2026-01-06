"""
-----------------------------------------------
1: Pre-Netoyage des données
-----------------------------------------------
Champs à supprimer:
    --------- nodes ----------
    osm_controller
    is_complete
    grid_region_id ?
    --------- links ----------
    osm_name
    osm_changeset
    startX
    startY
    endX
    endY
    birth_timestamp
    death_timestamp
    --------- travel_times_2013 ----------
    - Aucun -
Entrées à supprimer:
    --------- nodes ----------
    - Aucun -
    --------- links ----------
    osm_class:
        footway,
        platform,
        closed,
        proposed / construction
    begin_node_id orphelines
    end_node_id orphelines
    --------- travel_times_2013 ----------
    begin_node_id == end_node_id == 0
    begin_node_id and end_node_id not in links (tronçon de route introuvable)
-----------------------------------------------
2: Fetch depuis open_street_map
-----------------------------------------------
    
-----------------------------------------------
3: Post-netoyage
-----------------------------------------------
    
"""
import pandas as pd
from pathlib import Path


# Global variables
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"


nodesDf = pd.read_csv(RAW_DATA_DIR / "nodes.csv")
linksDf = pd.read_csv(RAW_DATA_DIR / "links.csv")
timesDf = pd.read_csv(RAW_DATA_DIR / "travel_times_2013.csv")

def pre_process_data():
    global nodesDf, linksDf, timesDf

    # suppression des champs inutiles

    nodesDf.drop(["osm_controller", "is_complete", "grid_region_id"], inplace=True)
    linksDf.drop([
        "osm_name",
        "osm_changeset",
        "startX",
        "startY",
        "endX",
        "endY",        
        "birth_timestamp",
        "death_timestamp"
    ], inplace= True)

    # suppression des entrees invalides

    # links
    osm_class_to_exclude = ['footway', 'platform', 'closed', 'proposed', 'construction']
    linksDf.drop(
        index =linksDf["osm_class"].isin[osm_class_to_exclude].index,
        inplace=True
    )

    primary_node_ids = set(nodesDf["node_id"].unique())

    invalid_links_mask = (
        ~linksDf["begin_node_id"].isin(primary_node_ids) |
        ~linksDf["end_node_id"].isin(primary_node_ids)
    )

    linksDf.drop(index=linksDf[invalid_links_mask].index, inplace=True)

    # travel_times
    invalid_times_mask = timesDf["begin_node_id"] == 0
    timesDf.drop(index=timesDf[invalid_times_mask].index, inplace=True)

    valid_links = linksDf[["begin_node_id", "end_node_id"]].drop_duplicates()

    timesDf = timesDf.merge(valid_links, on=["begin_node_id", "end_node_id"], how="inner")
    

def enrich_data_from_osm():
    global nodesDf, linksDf, timesDf

    api = overpy.Overpass()

