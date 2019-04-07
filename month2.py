import config
import helpers
import pandas as pd
config_ = config.Config()


curr, conn = helpers.connect_to_database()

#r = helpers.get_all_categories_from_db(curr, "business_az")
#helpers.get_frequency_list_of_categories(r)

'''
df_matched_points = helpers.yelp_osm_class_intersection(curr, 50, 'osm_pois_points')
df_matched_points = df_matched_points.set_index('business_id')
df_matched_polys = helpers.yelp_osm_class_intersection(curr, 50, 'osm_pois')
df_matched_polys = df_matched_polys.set_index('business_id')
resulting_df = df_matched_points.combine_first(df_matched_polys)
resulting_df.to_csv(config_.path+'\\resulting_df_matched_classes.csv')
print(resulting_df.shape)
'''

#convex_hull_string = helpers.get_convex_hull_as_text(curr, "business_az")
#buffered_convex_hull_geog = helpers.buffer_convex_hull(curr, convex_hull_string[0][0], 0.001)
# TODO change query since now we have geog instead of text in get_intersection_of_convex_hull_and_osm
#intersected_polygon = helpers.get_intersection_of_convex_hull_and_osm(curr, convex_hull_string[0][0], "osm_pois")

'''
df_matched_points = helpers.yelp_osm_name_intersection(curr, 50, 'osm_pois_points')
df_matched_points = df_matched_points.set_index('business_id')
df_matched_points['table_name'] = 'osm_pois_points'
df_matched_polys = helpers.yelp_osm_name_intersection(curr, 50, 'osm_pois')
df_matched_polys = df_matched_polys.set_index('business_id')
df_matched_polys['table_name'] = 'osm_pois'
resulting_df = df_matched_points.combine_first(df_matched_polys)
resulting_df.to_csv(config_.path+'\\resulting_df_matched_names.csv')
print(resulting_df.shape)
'''

# helpers.create_address_fields_in_osm_table(curr, conn, 'osm_pois_points')
# helpers.create_address_fields_in_osm_table(curr, conn, 'osm_pois')

helpers.update_address_fields_in_osm(curr, conn, 'D:\\ABH Internship\\dataset\\resulting_df_matched_names.csv')

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




