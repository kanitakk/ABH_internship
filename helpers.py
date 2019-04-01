import json
import pandas as pd
from pandas.io.json import json_normalize
import config
import time
import numpy as np
import psycopg2
import itertools
from fuzzywuzzy import fuzz
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


def get_distinct_values_from_db():
    conn = psycopg2.connect("dbname=dataset user=postgres password=kkk123 host=localhost")  #connect to db
    cur = conn.cursor()
    cur.execute("select distinct(categories) from business where categories is not null")   #get categories from db
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

    new_rows2 = list(set(new_rows2))  # after strip 1300 distinct categories
    return new_rows2


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


def drop_data_with_signficant_nulls(data):
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


def intersect_geometries(radius):
    conn = psycopg2.connect("dbname=dataset user=postgres password=kkk123 host=localhost")
    cur = conn.cursor()
    cur.execute("select az.business_id as yelp_id, az.name as yelp_name, az.categories, osm.osm_id as osm_id, osm.name as osm_name, osm.fclass from business_az as az, osm_pois as osm where ST_Intersects(ST_Buffer(az.geom, " + str(radius) +"), osm.geom) and osm.name is not null")
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=['yelp_id', 'yelp_name', 'categories', 'osm_id', 'osm_name', 'fclass'])
    grouped_yelp = df.groupby(by=['yelp_id'])
    return grouped_yelp


def fuzzy_match(grouped_df):
    # TODO try with other fuzzy functions and compare results
    matched = []
    scores_df = pd.DataFrame(columns=['business_id', 'yelp_name', 'osm_name', 'partial_ratio', 'ratio', 'sort_ratio', 'set_ratio'])
    for i, yelp in grouped_df:
        for n, row in yelp.iterrows():
            partial_ratio = fuzz.partial_ratio(row['yelp_name'], row['osm_name'])
            ratio = fuzz.ratio(row['yelp_name'], row['osm_name'])
            token_sort_ratio = fuzz.token_sort_ratio(row['yelp_name'], row['osm_name'])
            set_ration = fuzz.token_set_ratio(row['yelp_name'], row['osm_name'])

            if set_ration > 69:
                matched_pair = row['yelp_name'] + ' | ' + row['osm_name']
                # print(matched_pair)
                #print("partial ration: "+str(partial_ratio))
                #print("ration: "+str(ratio))
                #print("token_sort_ratio: "+str(token_sort_ratio))
                #print("fuzz_ration: "+str(set_ration))
                #print("\n")
                scores_df.loc[len(scores_df)] = i, row['yelp_name'], row['osm_name'], partial_ratio, ratio,\
                                                token_sort_ratio, set_ration
                matched.append(matched_pair)
    scores_df.to_csv(config_.path+'\\scores.csv', index=False)
    print(len(matched))


a = intersect_geometries(0.001)
fuzzy_match(a)