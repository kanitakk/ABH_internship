import json
import pandas as pd
import helpers
import config


config_ = config.Config()


flatten = helpers.normalize_json()
flatten = helpers.drop_empty_columns(flatten)
helpers.convert_df_to_csv(flatten, config_.path + "/reviewa_ALL.csv")
#flatten = pd.read_csv(config_.path+"\\complete.csv", sep="|",  low_memory=False)
#flatten = helpers.drop_empty_columns(flatten)
#flatten = flatten.rename(columns={col: col.lower() for col in flatten.columns})
#create_table_query = helpers.generate_fields_for_db(flatten, config_.sql_table_name)
# data = data.rename(columns={col: col.lower() for col in data.columns}) #convert columns to lowercase
'''
# function calls that help in attribute analysis

flatten = flatten.dropna(1, "all")     # delete columns that are completely empty
flatten.dropna(0, "any")               # check size of df which has complete data
descr = flatten.describe(include="all")
descr.to_csv("D:\\ABH Internship\\yelp_dataset\\description.csv")
counts = descr.loc["count"]
column_completeness = counts/len(flatten)*100   # calculate completeness of each column
freq_address = flatten["state"].value_counts()        # frequency of certain column
empty_count = len(flatten)-descr.loc["count"]           # empty values in columns
flatten.state.unique()      # example of dummy values using unique in column "state"
'''

