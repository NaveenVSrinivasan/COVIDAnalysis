from abstract_embed import *
from visualize_embeddings import *
import json


EMBEDDING_TYPE = "SCIBERT"
# EMBEDDING_TYPE = "SCIBERT_TF_IDF"

LOAD = False #Whether to build embeddings, or load them

USE_TITLES = False
WRITE_FILE = "scibert_embeds_all" #Place to save/load embeddings. 
abstracts = extract_abstracts('../data/biorxiv_medrxiv/biorxiv_medrxiv',remove_ints=True) #Directory from which to load abstracts
titles = extract_titles('../data/biorxiv_medrxiv/biorxiv_medrxiv') #Directory from which to load abstracts
		
# abstracts = extract_abstracts('../data/*/*',remove_ints=True) #Directory from which to load abstracts
# titles = extract_titles('../data/*/*') #Directory from which to load abstracts

if USE_TITLES:
    abstracts = titles

if LOAD:
    with open(WRITE_FILE,"r") as json_file:
        text_to_embeddings = json.load(json_file)
elif EMBEDDING_TYPE == "SCIBERT":
    text_to_embeddings = build_scibert_embeds(abstracts)
elif EMBEDDING_TYPE == "SCIBERT_TF_IDF":
    text_to_embeddings = build_scibert_embeds_tf_idf(abstracts)

# interval = 1000
# for start in range(0,10000,interval):
#     text_to_embeddings = build_scibert_embeds(abstracts[start:start+interval])

if not LOAD:
    for k in text_to_embeddings.keys(): #write embeddings to list so they can be stored as json
      text_to_embeddings[k] = text_to_embeddings[k].tolist()

    with open(str(start)+"_"+WRITE_FILE,"w+") as json_file: 
        json.dump(text_to_embeddings,json_file)

check_embeddings_for_NAN(text_to_embeddings) #remove any nans or infs in data

# run_elbow(text_to_embeddings)

text,clusters = visualize_embeddings(text_to_embeddings,reduce_fn="TSNE",write_to_file=True,file_name='embeddings_all') #visualize embeddings and write the reduced data to tsv file for visualizer
#To visualize, change file in visualizer to this file location

extract_cluster_names(text,clusters)
print_best_matches(text_to_embeddings) #for every abstract, print the best match

# num_clusters,value = calculate_num_clusters([v for _,v in text_to_embeddings.items()],kmax=15) #calculate the number of clusters using siloutte score
  
# print(num_clusters,value)



