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
    week_day:
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

