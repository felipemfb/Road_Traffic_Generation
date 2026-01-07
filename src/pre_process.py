"""
-----------------------------------------------
1: Pre-Netoyage des données
-----------------------------------------------
Champs à supprimer:
    --------- nodes ----------
    osm_controller
    is_complete
    grid_region_id
    birth_timestamp
    death_timestamp
    osm_changeset
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
Création de la table ways:
    way_id (depuis links)
    dest_node_id (si existe, sinon 0)
"""
import pandas as pd

def cleanup_data(linksDf, nodesDf, timesDf):
    """
    Nettoye les données incorrectes et inutiles
    """
    # suppression des champs inutiles

    nodesDf = nodesDf.drop([
        "is_complete",
        "osm_changeset",
        "osm_traffic_controller",
        "birth_timestamp",
        "death_timestamp",
        "grid_region_id"
        ], axis=1)

    linksDf = linksDf.drop([
        "osm_name",
        "osm_changeset",
        "startX",
        "startY",
        "endX",
        "endY",        
        "birth_timestamp",
        "death_timestamp"
    ], axis=1)

    # suppression des entrees invalides

    # links
    osm_class_to_exclude = ['footway', 'platform', 'closed', 'proposed', 'construction']
    linksDf = linksDf.drop(
        index=linksDf[linksDf["osm_class"].isin(osm_class_to_exclude)].index,
        axis = 1
    )

    primary_node_ids = set(nodesDf["node_id"].unique())

    invalid_links_mask = (
        ~linksDf["begin_node_id"].isin(primary_node_ids) |
        ~linksDf["end_node_id"].isin(primary_node_ids)
    )

    linksDf = linksDf.drop(index=linksDf[invalid_links_mask].index, axis = 1)

    # travel_times
    timesDf = timesDf.drop(["num_trips"], axis=1)
    
    invalid_times_mask = timesDf["begin_node_id"] == 0
    timesDf = timesDf.drop(index=timesDf[invalid_times_mask].index, axis=1)

    valid_links = linksDf[["begin_node_id", "end_node_id"]].drop_duplicates()

    timesDf = timesDf.merge(valid_links, on=["begin_node_id", "end_node_id"], how="inner")

    return linksDf, nodesDf, timesDf
    

def create_ways(linksDf):
    ways_data = []

    for way_id, group in linksDf.groupby("osm_way_id"):
        begin_nodes = set(group["begin_node_id"])
        end_nodes = set(group["end_node_id"])

        # si une fin de segment n'est pas le début d'un autre c'est la destination
        destination_nodes = end_nodes - begin_nodes

        if len(destination_nodes) == 1:
            dest_node_id = destination_nodes.pop()
            ways_data.append({
                "osm_way_id": way_id,
                "dest_node_id": dest_node_id
            })
        else: # On ne connait pas la destination
            dest_node_id = 0
            ways_data.append({
                "osm_way_id": way_id,
                "dest_node_id": dest_node_id
            })

    waysDf = pd.DataFrame(ways_data)

    return waysDf