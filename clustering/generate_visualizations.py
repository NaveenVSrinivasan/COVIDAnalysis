from argparse import ArgumentParser
import enum

from build_embedding_functions import *
from text_extraction_functions import *
from visualize_embedding_functions import *
import json


import sys


if len(sys.argv) != 3:
    print("Usage: python generate_visualizations.py embedding_file cluster_output_file")
    exit()


print("-------Loading Embeddings File-------")
with open(sys.argv[1], "r") as json_file:
    text_to_embeddings = json.load(json_file)

num_clusters = run_elbow(text_to_embeddings)

text, clusters = visualize_embeddings(text_to_embeddings, num_clusters=num_clusters, reduce_fn=Reduction.TSNE,
                                          write_to_file=True, file_name=sys.argv[2],show=False) 
extract_cluster_names(text, clusters)
  



