import pandas as pd
import gc

from pre_process import *
from enrich import *
from query_api import *

NEW_YORK_EPSG = 32618
CITY_NAME = "New York"
COUNTRY_NAME = "USA"

print("------------- loading CSVs -------------")
linksDf = pd.read_csv("data/raw/links.csv")
nodesDf = pd.read_csv("data/raw/nodes.csv")
timesDf = pd.read_csv("data/raw/travel_times_2013.csv")


print("############## Pre-processing ##############")

print("------------- cleaning data -------------")
linksDf, nodesDf, timesDf = cleanup_data(linksDf, nodesDf, timesDf)
gc.collect()

print("------------- creating table ways -------------")
waysDf = create_ways(linksDf)

print("############## Enriching ##############")

print("------------- getting ways tags -------------")
ways_tagsDf = get_ways(CITY_NAME)

print("------------- enriching table ways -------------")
waysDf = enrich_ways_max_speed_and_lanes(waysDf, ways_tagsDf)
gc.collect()

print(waysDf.shape)
print(waysDf.head)
print(waysDf.describe)