import pandas as pd
import gc

from pre_process import *
from enrich import *
from post_process import *
from query_api import *

NEW_YORK_EPSG = 32618
CITY_NAME = "New York"
COUNTRY_NAME = "USA"

print("loading CSVs")
linksDf = pd.read_csv("data/raw/links.csv")
nodesDf = pd.read_csv("data/raw/nodes.csv")
timesDf = pd.read_csv("data/raw/travel_times_2013.csv")


print("############## Pre-processing ##############")

print("cleaning data")
linksDf, nodesDf, timesDf = cleanup_data(linksDf, nodesDf, timesDf)
gc.collect()

print("creating table ways")
waysDf = create_ways(linksDf)
gc.collect()

print("############## Enriching ##############")

print("getting geodata")
gdf_landuse, gdf_place = get_geodata("New York", "USA")

print("enriching table nodes")

nodesGdf = convert_to_gdf(nodesDf, NEW_YORK_EPSG)
nodesGdf = enrich_nodes_with_tags(nodesGdf, gdf_landuse, gdf_place, NEW_YORK_EPSG)
nodesGdf = enrich_nodes_with_dist_to_center(nodesGdf)
gc.collect()

print("getting ways tags")
ways_tagsDf = get_ways(CITY_NAME)

print("enriching table ways")
waysDf = enrich_ways_max_speed_and_lanes(waysDf, ways_tagsDf)
gc.collect()

print("############## Post-processing ##############")

print("post processing table nodes")
nodesGdf = post_process_nodes(nodesGdf)
gc.collect()

print("post processing table ways")
waysDf = post_process_ways(linksDf, nodesGdf, waysDf)
gc.collect()

print("post processing table times")
timesDf = post_process_times(linksDf, timesDf)
gc.collect()

print("post processing table links")
linksDf = post_process_links(linksDf, nodesGdf, waysDf)
gc.collect()

print("creating final table")
trainingDf = create_training_table(linksDf, timesDf)
gc.collect()

print("encoding categorial features")
trainingDf = encode_categorial_features(trainingDf)
gc.collect()

print("writing table to data/processed/training.csv")
trainingDf.to_csv("data/processed/training.csv", index=False)