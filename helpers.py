import json
import pandas as pd
from pandas.io.json import json_normalize
import config
import time
import numpy as np
import psycopg2
import itertools
from fuzzywuzzy import fuzz
from nltk.stem import PorterStemmer
import collections
config_ = config.Config()


def flatten_json(y: dict) -> dict:
    """
    :param y:
    :return:
    """
    out = {}

    def flatten(x, name=''):
        if type(x) == str:
            try:
                x = x.replace("'", "\"")
                x = json.loads(x.lower())
            except:
                pass
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x
    flatten(y)
    return out


def normalize_json():
    temp_flatten = []
    start = time.time()
    i = 0
    for line in open(config_.path + "/review1.json", encoding="utf8"):
        temp_flatten.append(flatten_json(json.loads(line)))
        i += 1
        if i % 100000 == 0:
            print(len(temp_flatten))
            elapsed = time.time()-start
            print("Total time so far"+str(elapsed))
    a = json_normalize(temp_flatten)
    return a


def convert_df_to_csv(df, path):
    df.to_csv(path, sep="|", index=False)


def generate_fields_for_db(df, table_name):
    """

    :param df: dataframe
    :param table_name: string, table name which will be imported into query
    :return:
    """
    temp_dict = {}
    cols = df.columns
    query = "create table " + table_name + "("
    start = True
    for a in cols:
        non_nan_value = df[a].first_valid_index()
        if type(df[a][non_nan_value]) is str:
            temp_dict[a] = "varchar(1000)"
        elif type(df[a][non_nan_value]) is np.float64:
            temp_dict[a] = "float(48)"
        elif type(df[a][non_nan_value]) is np.int64:
            temp_dict[a] = "int"
        elif type(df[a][non_nan_value]) is bool:
            temp_dict[a] = "varchar(10)"
        else:
            print("Did not recognize data type in col: " + a)
    for key, value in temp_dict.items():
        if start:
            query += key + " " + value
            start = False
        else:
            query += ", " + key + " " + value

    query += ");"
    query = query.replace('-', '_')
    return query


def drop_empty_columns(flatten):
    flatten = flatten.dropna(axis=1, how="all")
    return flatten


def get_distinct_values_from_db(cur, table_name):
    cur.execute("select distinct(categories) from "+table_name+" where categories is not null")   #get categories from db
    rows = cur.fetchall()

    new_rows = []
    for r in rows:      # convert from tuple to list
        new_rows.append(list(r))

    new_rows = list(itertools.chain.from_iterable(new_rows))
    new_rows = ",".join(new_rows)
    new_rows = new_rows.lower()
    new_rows = new_rows.split(",")
    new_rows2 = []
    # without strip 2468 categories
    for n in new_rows:
        new_rows2.append(n.strip())     #strip white places

    new_rows2 = list(set(new_rows2))  # after strip 1300 distinct categories in USA, 1212 only in AZ
    return new_rows2


def get_all_categories_from_db(cur, table_name):
    cur.execute("select categories from " + table_name + " where categories is not null")  # get categories from db
    rows = cur.fetchall()

    new_rows = []
    for r in rows:  # convert from tuple to list
        new_rows.append(list(r))

    new_rows = list(itertools.chain.from_iterable(new_rows))
    new_rows = ",".join(new_rows)
    new_rows = new_rows.lower()
    new_rows = new_rows.split(",")
    new_rows2 = []
    # without strip
    for n in new_rows:
        new_rows2.append(n.strip())  # strip white places

    return new_rows2


def get_frequency_list_of_categories(categories_list):
    counter = collections.Counter(categories_list)
    print(counter)


def standardize_categories_and_classes(df, class_type):

    df[class_type] = df[class_type].str.lower()

    if class_type == 'osm_class2':
        for i, row in df.iterrows():
            for key, value in config_.standardized_dict_osm.items():
                if key in row[class_type]:
                    row[class_type] = row[class_type].replace(key, value)
        return df
    elif class_type == 'yelp_categories2':
        for i, row in df.iterrows():
            for key, value in config_.standardized_dict_yelp.items():
                if key in row[class_type]:
                    row[class_type] = row[class_type].replace(key, value)
        return df
    else:
        print("Wrong column name for categories.")


def categorize_data(data):
    # w_name = 20, w_address = 15, w_city = 10, w_postal_code = 5
    # w_state = 5, w_latitude = 10, w_longitude = 10, w_stars = 10, w_categories = 15
    # weights for columns, order of listed weights is important
    df_with_significant_cols = data[['name', 'address', 'city', 'postal_code', 'state', 'latitude', 'longitude', 'stars', 'categories']]
    df_is_null = df_with_significant_cols.isnull()      # mask for completeness
    df_weights = pd.DataFrame(index=df_is_null.index, columns=df_is_null.columns)
    df_weights['w_name'] = 20
    df_weights[df_weights.columns[1]] = 15  # address weight
    df_weights[df_weights.columns[2]] = 10  # city weight
    df_weights[df_weights.columns[3]] = 5  # postal_code weight
    df_weights[df_weights.columns[4]] = 5  # state weight
    df_weights[df_weights.columns[5]] = 10  # latitude weight
    df_weights[df_weights.columns[6]] = 10  # longitude weight
    df_weights[df_weights.columns[7]] = 10  # stars weight
    df_weights[df_weights.columns[8]] = 15  # categories weight
    scores_df = df_weights * ~df_is_null
    total_scores = scores_df.sum(axis=1)
    num_of_low_quality = len(total_scores.loc[total_scores < 46])
    print("Percentage of low quality data is: " + str(num_of_low_quality * 100 / 192609))
    num_of_middle_quality = len(total_scores.loc[(total_scores > 45) & (total_scores < 80)])
    print("Percentage of middle quality data is: " + str(num_of_middle_quality * 100 / 192609))
    num_of_high_quality = len(total_scores.loc[total_scores > 79])
    print("Percentage of high quality data is: " + str(num_of_high_quality * 100 / 192609))
    # df1.loc[df1['stream'] == 2, ['feat','another_feat']] = 'aaaa' changing value of cell conditionally


def drop_data_with_significant_nulls(data):
    df_with_significant_cols = data[['name', 'address', 'city', 'postal_code', 'state', 'latitude', 'longitude', 'categories']]
    df_without_significant_nulls = df_with_significant_cols.dropna(axis=0)
    num_of_low_quality = 192609 - len(df_without_significant_nulls)
    print("Percentage of low quality " + str((num_of_low_quality+25)*100/192609))


def geocode_data():
    import requests
    import json
    import pandas as pd
    import numpy as np
    from threading import Thread
    from queue import Queue

    KEY = ""

    df = pd.read_csv("D:\\ABH Internship\\yelp_dataset\\small.csv")
    df['full_address'] = df['address'] + ', ' + df['city'] + ' ' + df['state'] + ' ' + df['postal_code']
    addresses = df['full_address']
    address_list = np.array(addresses)
    index_list = np.array(addresses.index.tolist())

    location = pd.DataFrame(columns=['Id', 'Adresa', 'Lat', 'Lng'], index=np.arange(0, len(addresses)))
    num = 0
    for i in range(len(addresses)):
        r = requests.get("https://maps.googleapis.com/maps/api/geocode/json?address=Arizona," + str(
            addresses.Adresa.loc[i]) + "&key=" + KEY)
        json_string = r.content.decode()
        json_object = json.loads(json_string)
        status = json_object['status']
        if status == 'OK':
            lat = json_object['results'][0]['geometry']['location']['lat']
            lng = json_object['results'][0]['geometry']['location']['lng']
            location.Id.loc[i] = i
            location.Adresa.loc[i] = addresses.Adresa.loc[i]
            location.Lat.loc[i] = lat
            location.Lng.loc[i] = lng
        print(num)
        num += 1

    location.to_csv("D:\\ABH Internship\\yelp_dataset\\coordinatesFromAPI.csv")


def stemming_match(row, ps, temp_yelp_lista, temp_osm_lista, yelp_column, osm_columns):
    yelp_list = [row[yelp_column]]  # prebaci u listu kompletan string
    yelp_list = yelp_list[0].split(",")  # splita po zarezu
    yelp_list = [item.strip() for item in yelp_list]
    yelp_list = [" ".join(yelp_list)] # splituje po space da bi dobili posebno svaku rijec
    yelp_list = yelp_list[0].split(" ")
    for word in yelp_list:
        temp_yelp_lista.append(ps.stem(word) + " ")

    osm_list = [row[osm_columns]]
    osm_list = [item.strip() for item in osm_list]
    for word in osm_list:
        temp_osm_lista.append(ps.stem(word) + " ")


def yelp_osm_name_intersection(curr, meters, table_name):
    intersected_geographies = intersect_geometries(curr, meters, table_name)
    intersected_geographies = pd.DataFrame(intersected_geographies,
                                           columns=['yelp_id', 'yelp_name', 'categories', 'osm_id', 'osm_name',
                                                    'fclass'])
    intersected_geographies = normalize_names(intersected_geographies, "osm_name", "yelp_name")
    print('There are '+str(len(intersected_geographies))+' intersected geographies in '+table_name)
    scores_df = fuzzy_match(intersected_geographies, table_name)
    return scores_df


def yelp_osm_class_intersection(curr, radius_in_meters, table_name):
    rows = intersect_geometries_without_osm_name(curr, radius_in_meters, 'osm_pois_points')
    print('There are '+str(len(rows))+'intersected geometries with '+table_name)
    if table_name == 'osm_pois_points':
        destination = 'standardized_points'
    elif table_name == 'osm_pois':
        destination = 'standardized_polys'
    matched_df = fuzzy_match_standardized_classes(rows, destination)
    return matched_df


def fuzzy_match(rows, table_name):
    print("Doing fuzzy matching for names")
    matched = []
    df = pd.DataFrame(rows, columns=['yelp_id', 'yelp_name', 'categories', 'osm_id', 'osm_name', 'fclass'])
    grouped_yelp = df.groupby(by=['yelp_id'])
    ps = PorterStemmer()
    scores_df = pd.DataFrame(columns=['business_id', 'osm_id', 'yelp_name', 'stemmed_yelp', 'osm_name', 'stemmed_osm',
                                      'partial_ratio', 'ratio', 'sort_ratio', 'set_ratio'])
    for i, yelp in grouped_yelp:
        for n, row in yelp.iterrows():
            temp_yelp_lista = []
            temp_osm_lista = []
            stemming_match(row, ps, temp_yelp_lista, temp_osm_lista, "yelp_name", "osm_name")

            temp_yelp_lista = " ".join(temp_yelp_lista)
            temp_yelp_lista = temp_yelp_lista.strip()
            temp_osm_lista = " ".join(temp_osm_lista)
            temp_osm_lista = temp_osm_lista.strip()

            partial_ratio = fuzz.partial_ratio(temp_yelp_lista, temp_osm_lista)
            ratio = fuzz.ratio(temp_yelp_lista, temp_osm_lista)
            token_sort_ratio = fuzz.token_sort_ratio(temp_yelp_lista, temp_osm_lista)
            set_ration = fuzz.token_set_ratio(temp_yelp_lista, temp_osm_lista)

            if set_ration > 69:
                matched_pair = row['yelp_name'] + ' | ' + row['osm_name']
                scores_df.loc[len(scores_df)] = i, row['osm_id'], row['yelp_name'], temp_yelp_lista, row['osm_name'],\
                                                temp_osm_lista, partial_ratio, ratio, token_sort_ratio, set_ration
                matched.append(matched_pair)
    scores_df.to_csv(config_.path+'\\scores_with_stemming'+table_name+'.csv', index=False, encoding="latin-1")
    print(len(matched))
    return scores_df


def connect_to_database():
    print("Connecting to database")
    conn = psycopg2.connect("dbname="+config_.db_name+" user="+config_.db_user+" password="+config_.db_password+" host=localhost")
    cur = conn.cursor()
    return cur


def intersect_geometries(cur, radius_in_meters, table_name):
    radius = radius_in_meters*0.001/111
    print("Intersecting osm with yelp points")
    cur.execute("select az.business_id as yelp_id, az.name as yelp_name, az.categories, osm.osm_id as osm_id,"
                " osm.name as osm_name, osm.fclass from business_az as az, "
                +table_name+" as osm where ST_Intersects(ST_Buffer(az.geom, " + str(radius) +"), osm.geom) "
                "and osm.name is not null")
    rows = cur.fetchall()
    return rows


def get_convex_hull_as_text(curr, table):
    curr.execute("SELECT ST_AsEWKT(ST_ConvexHull(ST_Collect(" + table + ".geom))) AS convex_hull FROM " + table + ";")
    convex_hull_string = curr.fetchall()
    return convex_hull_string


def get_intersection_of_convex_hull_and_osm(curr, convex_hull_string, table):
    curr.execute("select "+table+".osm_id from "+table+" where ST_Intersects(ST_GeomFromText('" + convex_hull_string + "'), "+table+".geom)")
    osm_ids = curr.fetchall()
    return osm_ids


def delete_specific_osm_classes(curr, table):
    query = "delete from "+table+" where fclass in ('waste_basket', 'toilet')"
    curr.execute(query)
    geom = curr.fetchall()
    return geom


def buffer_convex_hull(curr, tex_polygon, radius_in_meters):
    radius = radius_in_meters*0.001/111
    query = "select ST_AsEWKT(ST_Buffer(ST_GeomFromText('" + tex_polygon +"'), "+str(radius)+"))"
    curr.execute(query)
    buffered_polygon = curr.fetchall()
    return buffered_polygon


def normalize_names(df, osm_col, yelp_col):
    print("Normalizing name columns")
    df[osm_col] = df[osm_col].astype(str)
    df[yelp_col] = df[yelp_col].astype(str)
    df = to_lower(df, osm_col, yelp_col)
    df = strip_names(df, osm_col, yelp_col)
    df = remove_diacritics(df, osm_col, yelp_col)
    df = replace_special_characters(df, osm_col, yelp_col)
    return df


def remove_diacritics(df, osm_col, yelp_col):

    df[osm_col] = df[osm_col].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    df[yelp_col] = df[yelp_col].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    return df


def replace_special_characters(df, osm_col, yelp_col):
    df[osm_col] = df[osm_col].str.replace("'", "")
    df[osm_col] = df[osm_col].str.replace('"', "")
    df[yelp_col] = df[yelp_col].str.replace("'", "")
    df[yelp_col] = df[yelp_col].str.replace('"', "")

    # removing right slash
    df[osm_col] = df[osm_col].str.replace(' / ', " ")
    df[yelp_col] = df[yelp_col].str.replace(' / ', " ")
    df[osm_col] = df[osm_col].str.replace('/ ', " ")
    df[yelp_col] = df[yelp_col].str.replace('/ ', " ")
    df[osm_col] = df[osm_col].str.replace(' /', " ")
    df[yelp_col] = df[yelp_col].str.replace(' /', " ")
    df[osm_col] = df[osm_col].str.replace('/', " ")
    df[yelp_col] = df[yelp_col].str.replace('/', " ")

    # replacing exclamation mark
    df[osm_col] = df[osm_col].str.replace('!', "")
    df[yelp_col] = df[yelp_col].str.replace('!', "")

    # replacing dots
    df[osm_col] = df[osm_col].str.replace('.', "")
    df[yelp_col] = df[yelp_col].str.replace('.', "")

    # replacing dash
    df[osm_col] = df[osm_col].str.replace(' - ', " ")
    df[yelp_col] = df[yelp_col].str.replace(' - ', " ")
    df[osm_col] = df[osm_col].str.replace('- ', " ")
    df[yelp_col] = df[yelp_col].str.replace('- ', " ")
    df[osm_col] = df[osm_col].str.replace(' -', " ")
    df[yelp_col] = df[yelp_col].str.replace(' -', " ")
    df[osm_col] = df[osm_col].str.replace('-', " ")
    df[yelp_col] = df[yelp_col].str.replace('-', " ")

    # replacing plus
    try:
        df[osm_col] = df[osm_col].str.replace(' + ', " ")
        df[yelp_col] = df[yelp_col].str.replace(' + ', " ")
        df[osm_col] = df[osm_col].str.replace('+ ', " ")
        df[yelp_col] = df[yelp_col].str.replace('+ ', " ")
        df[osm_col] = df[osm_col].str.replace(' +', " ")
        df[yelp_col] = df[yelp_col].str.replace(' +', " ")
        df[osm_col] = df[osm_col].str.replace('+', " ")
        df[yelp_col] = df[yelp_col].str.replace('+', " ")
    except:
        df[osm_col] = df[osm_col].str.replace('+', " ")
        df[yelp_col] = df[yelp_col].str.replace('+', " ")

    # removing underscore
    df[osm_col] = df[osm_col].str.replace('_', " ")
    df[yelp_col] = df[yelp_col].str.replace('_', " ")

    return df


def strip_names(df, osm_col, yelp_col):
    df[osm_col] = df[osm_col].str.strip()
    df[yelp_col] = df[yelp_col].str.strip()
    return df


def to_lower(df, osm_col, yelp_col):
    df[osm_col] = df[osm_col].str.lower()
    df[yelp_col] = df[yelp_col].str.lower()
    return df


def fuzzy_match_classes(rows):
    print("Doing fuzzy matching for categories/classes")
    matched = []
    df = pd.DataFrame(rows, columns=['yelp_id', 'osm_id', 'yelp_categories', 'osm_class'])
    grouped_yelp = df.groupby(by=['yelp_id'])
    ps = PorterStemmer()
    scores_df = pd.DataFrame(columns=['business_id', 'yelp_categories', 'osm_class', 'fuzz_ratio'])
    for i, yelp in grouped_yelp:
        temp_yelp_lista = []
        temp_osm_lista = []
        for n, row in yelp.iterrows():
            yelp_list = [row['yelp_categories']] #prebaci u listu kompletan string
            yelp_list = yelp_list[0].split(",") # splita po zarezu
            yelp_list = [item.strip() for item in yelp_list] # stripuje sve te vrijednosti da ostane "game" a ne " game"
            yelp_list = [" ".join(yelp_list)] # splituje po space da bi dobili posebno svaku rijec

            for word in yelp_list:
                temp_yelp_lista.append(ps.stem(word)+" ")

            osm_list = [row['osm_class']]
            osm_list = [item.strip() for item in osm_list]
            for word in osm_list:
                temp_osm_lista.append(ps.stem(word)+" ")

        temp_yelp_lista = " ".join(temp_yelp_lista)
        temp_yelp_lista = temp_yelp_lista.strip()
        temp_osm_lista = " ".join(temp_osm_lista)
        temp_osm_lista = temp_osm_lista.strip()

        fuzz_ratio = fuzz.token_set_ratio(temp_osm_lista, temp_yelp_lista)
        if fuzz_ratio > 70:
            matched_pair = row['yelp_categories'] + ' | ' + row['osm_class']
            scores_df.loc[len(scores_df)] = i, row['yelp_categories'], row['osm_class'], fuzz_ratio
            matched.append(matched_pair)
    scores_df.to_csv(config_.path+'\\classes_scores.csv', index=False, encoding="latin-1")
    print("Number of matched classes: " + str(len(matched)))


def fuzzy_match_standardized_classes(rows, destination_name):
    print("Doing fuzzy matching for standardized categories/classes")
    matched = []
    df = pd.DataFrame(rows, columns=['yelp_id', 'osm_id', 'yelp_name', 'yelp_categories', 'osm_class'])
    df = df[:500]
    df['osm_class2'] = df['osm_class']
    df['yelp_categories2'] = df['yelp_categories']
    grouped_yelp = df.groupby(by=['yelp_id'])
    scores_df = pd.DataFrame(columns=['business_id', 'yelp_name', 'yelp_categories2', 'osm_class2', 'fuzz_ratio'])
    for i, yelp in grouped_yelp:
        yelp = standardize_categories_and_classes(yelp, 'yelp_categories2')
        yelp = standardize_categories_and_classes(yelp, 'osm_class2')
        for n, row in yelp.iterrows():
            fuzz_ratio = fuzz.token_set_ratio(row['osm_class2'], row['yelp_categories2'])
            if fuzz_ratio > 90:
                matched_pair = row['yelp_categories2'] + ' | ' + row['osm_class2']
                scores_df.loc[len(scores_df)] = i, row['yelp_name'], row['yelp_categories2'], row['osm_class2'], fuzz_ratio
                matched.append(matched_pair)
    scores_df.to_csv(config_.path+'\\scores'+destination_name+'.csv', index=False, encoding="latin-1")
    print("Number of matched classes: " + str(len(matched)))
    return scores_df


def intersect_geometries_without_osm_name(cur, radius_in_meters, table_name):
    print("Intersecting yelp with "+table_name+" without osm name")
    radius = radius_in_meters*0.001/111
    cur.execute("select az.business_id as yelp_id, osm.osm_id as osm_id, az.name as yelp_name, "
                "az.categories as yelp_categories, osm.fclass as osm_class from business_az as az, "
                +table_name+" as osm where ST_Intersects(ST_Buffer(az.geom, " + str(radius) +"), osm.geom) "
                "and osm.name is null and az.categories is not null")
    rows = cur.fetchall()
    return rows
