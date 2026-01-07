"""
-----------------------------------------------
3: Post-netoyage
-----------------------------------------------
---------- Données à transformer: -----------
Nodes:
    place:
        0: town 1: quarter 2: neighborhood/cityblock 3:plot 4:village 5:hamlet 6:allotments/dwellings/farm
    centrality:
        normalisation z-score
    num_links:
        à partir de num_in_links et num_out_links

links:
    angle:
        calcul à partir de begin_angle et end_angle

ways:
    max_speed:
        conversion en kmh puis en int

travel_times:
    day_time:
        à partir de datetime
    is_weekend:
        à partir de datetime

---------- Données à recouper: -----------
ways:
    landuse_dest:
        si dest != 0: prendre landuse de dest
        sinon: inférer par rapport aux nodes
    place_dest:
        idem landuse_dest

links:
    landuse, place, centrality, num_begin_links, num_out_links: à partir de begin et end_node (stratégie pour choisir parmis les deux ???)
    landuse_way_dest, place_way_dest, max_speed, num_lanes: join avec ways

travel_times:
    inférer link_id à partir de begin_node_id  end_node_id

---------- Données à supprimer: -----------
links:
    begin_node_id', 'end_node_id', 'begin_angle', 'end_angle', 'way_id'

travel_times:
    entrées num_trips < seuil ?? puis colonne num_trips


---------- Données finales: -----------
links:
    link_id, landuse, landuse_way_dest, place, place_way_dest, angle, max_speed, num_lanes, centrality, num_begin_links, num_out_links, road_type
travel_times:
    link_id, day_time, week_day
"""

import pandas as pd
import geopandas as gpd
import re

###########################################################
# Transformations
###########################################################

# nodes

def _map_place_to_int(nodesGdf):
    place_map = {
        "town": 0,
        "quarter": 1,
        "neighbourhood": 2, "cityblock": 2,
        "plot": 3,
        "village": 4,
        "hamlet": 5,
        "allotments": 6, "dwellings": 6, "farm": 6
    }

    nodesGdf["place"] = (nodesGdf["place"].str.lower().map(place_map).fillna(-1).astype(int))

    return nodesGdf

def _standardize_dist_to_center(nodesGdf):
    mu = nodesGdf["dist_to_center"].mean()
    sigma = nodesGdf["dist_to_center"].std()

    nodesGdf["centrality_z"] = (nodesGdf["dist_to_center"] - mu) / sigma

def _add_num_links(nodesGdf):
    nodesGdf["num_links"] = nodesGdf["num_in_links"] + nodesGdf["num_out_links"]

    return nodesGdf

# links

def _add_angle(linksDf):
    linksDf["angle"] =  abs(linksDf["begin_angle"] - linksDf["end_angle"])
    linksDf["angle"] = min(linksDf["angle"], 360 - linksDf["angle"])

    return linksDf

# ways

# Fait par ia me demandez pas
def _parse_maxspeed_to_kmh(maxspeed):
    """
    Convertit un tag OSM maxspeed en km/h (float).
    Retourne None si non exploitable.
    """
    if maxspeed is None or not isinstance(maxspeed, str):
        return None

    maxspeed = maxspeed.lower().strip()

    # valeurs non numériques connues
    if maxspeed in {"signals", "walk", "none", "variable"}:
        return None

    # prendre la première valeur si "50;70"
    maxspeed = maxspeed.split(";")[0].strip()

    # extraire le nombre
    match = re.search(r"(\d+(\.\d+)?)", maxspeed)
    if not match:
        return None

    value = float(match.group(1))

    # conversion selon unité
    if "mph" in maxspeed:
        return value * 1.60934
    elif "knot" in maxspeed:
        return value * 1.852
    else:
        # par défaut km/h
        return value
    

def _convert_max_speed(waysDf):
    waysDf["max_speed_kmh"] = waysDf["maxspeed"].apply(_parse_maxspeed_to_kmh)

    return waysDf


# travel_times
def _convert_datetime_type(timesDf):
    timesDf["datetime"] = pd.to_datetime(timesDf["datetime"])
    return timesDf

def _add_day_time(timesDf):
    timesDf["hour"] = timesDf["datetime"].dt.hour
    return timesDf


def _add_week_day(timesDf):
    timesDf["is_weekend"] = timesDf["datetime"].dt.weekday >= 5
    return timesDf

