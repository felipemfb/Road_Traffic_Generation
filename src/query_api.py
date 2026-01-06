"""
Service communiquant avec l'api d'osm
"""
import osmnx as ox
import overpy 
import pandas as pd
import time

def get_geodata(city: str, country: str):
    try:
        gdf_landuse = ox.features_from_place(f"{city}, {country}", tags={"landuse": True})
        gdf_place = ox.features_from_place(f"{city}, {country}", tags={"place": True})
    except Exception as e:
        print("failed to query gdf with osmnx")

    return gdf_landuse, gdf_place

def get_ways(city_name, max_attempts=10, delay=2):
    api = overpy.Overpass()

    query = f"""
    [out:json][timeout:180];
    area["name"="{city_name}"]["boundary"="administrative"]->.searchArea;
    way["highway"](area.searchArea);
    out tags;
    """
    retries = 0
    while retries <= max_attempts:
        try:
            result = api.query(query)
            ways_data = []
            for way in result.ways:
                lanes = way.tags.get("lanes")
                if lanes is None:
                    lanes = way.tags.get("lanes:forward")
                if lanes is not None:
                    # prendre juste le premier nombre si plusieurs séparés par ";"
                    lanes = str(lanes).split(";")[0]
                    try:
                        lanes = int(lanes)
                    except ValueError:
                        lanes = None

                ways_data.append({
                    "osm_way_id": int(way.id),
                    "maxspeed": way.tags.get("maxspeed"),
                    "lanes": lanes
                })

            return pd.DataFrame(ways_data)
        except (overpy.exception.OverpassGatewayTimeout,
            overpy.exception.OverpassTooManyRequests,
            overpy.exception.OverpassBadRequest) as e:
            retries += 1
            wait_time = delay * retries
            print(f"[WARN] Overpass API error: {e}. Retry {retries}/{max_attempts} after {wait_time}s...")
            time.sleep(wait_time)
    print("Failed to fetch ways from overpy, max_attempts passed")
    return []
            
