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


---------- Données finales: -----------
links:
    link_id, landuse, landuse_way_dest, place, place_way_dest, angle, max_speed, num_lanes, centrality, num_begin_links, num_out_links, road_type
travel_times:
    link_id, day_time, week_day, speed

training:
    landuse, landuse_way_dest, place, place_way_dest, angle, max_speed, num_lanes, centrality, num_begin_links, num_out_links, road_type, speed(label)
"""

import pandas as pd
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

    nodesGdf["centrality"] = (nodesGdf["dist_to_center"] - mu) / sigma

    return nodesGdf

def _add_num_links(nodesGdf):
    nodesGdf["num_links"] = nodesGdf["num_in_links"] + nodesGdf["num_out_links"]

    return nodesGdf

# links

def _add_angle(linksDf):
    linksDf["angle"] =  abs(linksDf["begin_angle"] - linksDf["end_angle"])
    linksDf["angle"] = linksDf["angle"].apply(lambda x: min(x, 360 - x))

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
    timesDf["is_weekend"] = (timesDf["datetime"].dt.weekday >= 5).astype(int)
    return timesDf


###########################################################
# Jointures
###########################################################

# vibe code mais ça a l'air ok
def _infer_landuse_and_place_dest(linksDf, nodesGdf, waysDf):
    # Indexes pour accélérer
    node_landuse = nodesGdf.set_index("node_id")["landuse"]
    node_place = nodesGdf.set_index("node_id")["place"]

    # mapping des destinations connues
    waysDf["landuse_dest"] = waysDf["dest_node_id"].map(node_landuse)
    waysDf["place_dest"] = waysDf["dest_node_id"].map(node_place)

    # récupération des nodes par way
    way_nodes = (
        linksDf
        .melt(
            id_vars="osm_way_id",
            value_vars=["begin_node_id", "end_node_id"],
            value_name="node_id"
        )
        [["osm_way_id", "node_id"]]
        .drop_duplicates()
    )

    way_nodes = way_nodes.merge(
        nodesGdf[["node_id", "landuse", "place"]],
        on="node_id",
        how="left"
    )

    def infer_landuse(series):
        series = series[series != "unknown"]
        return series.mode().iloc[0] if not series.empty else "unknown"

    def infer_place(series):
        series = series[series >= 0]
        return series.min() if not series.empty else -1

    way_inferred = (
        way_nodes
        .groupby("osm_way_id")
        .agg(
            landuse_inf=("landuse", infer_landuse),
            place_inf=("place", infer_place)
        )
        .reset_index()
    )

    waysDf = waysDf.merge(way_inferred, on="osm_way_id", how="left")

    waysDf["landuse_dest"] = (
        waysDf["landuse_dest"]
        .fillna(waysDf["landuse_inf"])
        .fillna("unknown")
    )

    waysDf["place_dest"] = (
        waysDf["place_dest"]
        .fillna(waysDf["place_inf"])
        .fillna(-1)
    )

    waysDf = waysDf.drop(["landuse_inf", "place_inf"], axis=1)

    return waysDf

def _join_links(linksDf, nodesGdf, waysDf):
    # join avec nodes
    nodes_lookup = nodesGdf.set_index("node_id")[["landuse", "place", "centrality", "num_links"]]

    linksDf = linksDf.merge(
        nodes_lookup,
        left_on="begin_node_id",
        right_index=True,
        how="left"
    )

    # Choix des colonnes
    linksDf["num_begin_links"] = linksDf["num_links"]

    linksDf = linksDf.drop(["num_links"], axis=1)

    # merge on end_node_id pour récupérer num_out_links

    linksDf = linksDf.merge(
        nodes_lookup[["num_links"]],
        left_on="end_node_id",
        right_index=True,
        how="left"
    )
    linksDf["num_out_links"] = linksDf["num_links"]
    linksDf = linksDf.drop(columns=["num_links"], axis=1)

    # join avec ways
    linksDf = linksDf.merge(
        waysDf[["osm_way_id", "landuse_dest", "place_dest", "max_speed_kmh", "lanes"]],
        left_on="osm_way_id",
        right_on="osm_way_id",
        how="left"
    )

    linksDf = linksDf.rename(columns={
        "landuse_dest": "landuse_way_dest",
        "place_dest": "place_way_dest",
        "max_speed_kmh": "max_speed",
    })

    linksDf = linksDf.drop(["osm_way_id"], axis=1)

    return linksDf

def _complete_max_speeds(linksDf):
    road_speed_map = {
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
        "services": 20,
        "road": 30,
        "living_street": 10,
        "raceway": 120,
        "1stStreet": 30,
    }

    mask = linksDf["max_speed"].isna()
    linksDf.loc[mask, "max_speed"] = linksDf.loc[mask, "osm_class"].map(road_speed_map)

    return linksDf

def _complete_times_from_links(linksDf, timesDf):
    links_lookup = linksDf[["link_id", "begin_node_id", "end_node_id", "street_length"]]

    timesDf = timesDf.merge(
        links_lookup,
        on=["begin_node_id", "end_node_id"],
        how="left"
    )

    timesDf["speed"] = (timesDf["street_length"] / timesDf["travel_time"]) * 3.6

    timesDf = timesDf.drop(["travel_time", "street_length", "begin_node_id", "end_node_id"], axis=1)
    return timesDf

###########################################################
# Netoyage des champs
###########################################################

def _cleanup_links_fields(linksDf):
    linksDf = linksDf.drop(['begin_node_id', 'end_node_id', 'begin_angle', 'end_angle'], axis=1)

    return linksDf


###########################################################
# Pipeline functions
###########################################################

def post_process_nodes(nodesGdf):
    nodesGdf = _map_place_to_int(nodesGdf)
    nodesGdf = _standardize_dist_to_center(nodesGdf)
    nodesGdf = _add_num_links(nodesGdf)

    return nodesGdf

def post_process_ways(linksDf, nodesGdf, waysDf):
    waysDf = _convert_max_speed(waysDf)
    waysDf = _infer_landuse_and_place_dest(linksDf, nodesGdf, waysDf)

    return waysDf

def post_process_times(linksDf, timesDf):
    timesDf = _convert_datetime_type(timesDf)
    timesDf = _add_day_time(timesDf)
    timesDf = _add_week_day(timesDf)
    timesDf = _complete_times_from_links(linksDf, timesDf)
    
    return timesDf

def post_process_links(linksDf, nodesGdf, waysDf):
    linksDf = _join_links(linksDf, nodesGdf, waysDf)
    linksDf = _complete_max_speeds(linksDf)
    linksDf = _add_angle(linksDf)
    linksDf = _cleanup_links_fields(linksDf)

    return linksDf



def create_training_table(linksDf, timesDf):
    timesDf = timesDf[['link_id', 'hour', 'is_weekend', 'speed']]

    timesAgg = timesDf.groupby(['link_id', 'hour', 'is_weekend'], as_index=False).agg(
        speed=('speed', 'mean')
    )

    linksDf = linksDf.rename(columns={"osm_class": "road_type"})

    # Merge
    links_with_times = linksDf.merge(
        timesAgg,
        on='link_id',
        how='inner'  # on ne garde que les links avec travel_times disponibles (mais normalement déjà géré au preprocess)
    )
    
    
    links_with_times = links_with_times.drop(["link_id"], axis=1)
    return links_with_times

def encode_categorial_features(trainingDf):
    trainingDf = pd.get_dummies(trainingDf, columns=[
        'road_type',
        'landuse',
        'landuse_way_dest'
    ])

    return trainingDf