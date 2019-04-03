import config
import helpers
import pandas as pd
config_ = config.Config()


curr = helpers.connect_to_database()

r = helpers.get_all_categories_from_db(curr, "business_az")
helpers.get_frequency_list_of_categories(r)

#convex_hull_string = helpers.get_convex_hull_as_text(curr, "business_az")
#buffered_convex_hull_geog = helpers.buffer_convex_hull(curr, convex_hull_string[0][0], 0.001)
# TODO change query since now we have geog instead of text in get_intersection_of_convex_hull_and_osm
#intersected_polygon = helpers.get_intersection_of_convex_hull_and_osm(curr, convex_hull_string[0][0], "osm_pois")
#intersected_geographies = helpers.intersect_geometries(curr, 0.001)
#intersected_geographies = pd.DataFrame(intersected_geographies, columns=['yelp_id', 'yelp_name', 'categories', 'osm_id', 'osm_name', 'fclass'])
#intersected_geographies = helpers.normalize_names(intersected_geographies, "osm_name", "yelp_name")
#print(len(intersected_geographies))
#helpers.fuzzy_match(intersected_geographies)

#yelp_convex_hull = helpers.get_convex_hull_as_text(curr, 'business_az')
#buffered_yelp_convex_hull = helpers.buffer_convex_hull(curr, yelp_convex_hull[0][0], 0.001)
#osm_points_in_yelp = helpers.get_intersection_of_convex_hull_and_osm(curr, buffered_yelp_convex_hull[0][0], 'osm_pois_points')
#osm_polys_in_yelp = helpers.get_intersection_of_convex_hull_and_osm(curr, buffered_yelp_convex_hull[0][0], 'osm_pois')
#print("There are "+str(len(osm_points_in_yelp))+" intersected osm points with yelp")
#print("There are "+str(len(osm_polys_in_yelp))+" intersected osm polygons with yelp")

#rows = helpers.intersect_geometries_without_osm_name(curr, 0.001)
#rows = pd.DataFrame(rows, columns=['yelp_id', 'yelp_name', 'yelp_categories', 'osm_id', 'osm_class'])
#rows = helpers.normalize_names(rows, "osm_class", "yelp_categories")
#print(len(rows))
#helpers.fuzzy_match_classes(rows)




