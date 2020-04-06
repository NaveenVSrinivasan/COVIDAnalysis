import pandas as pd
import sqlite3
import json

# Connect to database
db = sqlite3.connect("/Users/emilymu/Documents/COVIDAnalysis/clustering/articles.sqlite")

print("Loading Data")

# Articles
pd.set_option("max_colwidth", 125)
articles = pd.read_sql_query("select source, published, publication, authors, title, tags, loe, reference, "
                             "id from articles", db)

print("Articles Sample: ")
print(articles.head)

# Sections
sections = pd.read_sql_query("select * from sections", db)
sections = sections.sort_values(by=['Id'])

print("Sections Sample: ")
print(sections.head)

print("Creating Sections File: ")
sections_dict = {}

for index, row in sections.iterrows():
    key = (row['Article'], row['Name'])
    if key in sections_dict:
        sections_dict[key].append(row['Text'])
    else:
        sections_dict[key] = [row['Text']]

print("Keys: ")
print(len(sections_dict.keys()))

print("Creating Title to Abstract Mapping: ")
title_to_abstract_dict = {}

for index, row in articles.iterrows():
    if (row['Id'], 'TITLE') in sections_dict and (row['Id'], 'ABSTRACT') in sections_dict:
        title_to_abstract_dict[sections_dict[(row['Id'], 'TITLE')][0]] = sections_dict[(row['Id'], 'ABSTRACT')]

print("Keys: ")
print(len(title_to_abstract_dict.keys()))

print("Saving Files: ")

with open("all_sections.json", "w+") as json_file:
    json.dump({str(k):v for k, v in sections_dict.items()}, json_file)

with open("title_to_abstract_mapping.json", "w+") as json_file:
    json.dump(title_to_abstract_dict, json_file)



