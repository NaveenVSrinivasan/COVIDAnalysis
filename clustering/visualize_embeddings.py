import os          
import glob    
import json      
import numpy as np
import nltk.data
from tqdm import tqdm
import matplotlib.pyplot as plt
from  sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from numpy import inf

def reduce_dims(embeddings,type="TSNE"):
    if type == "TSNE":
        X_embedded = TSNE(n_components=2).fit_transform(embeddings)
    elif type == "PCA":
        X_embedded = PCA(n_components=2).fit_transform(embeddings)

    x = X_embedded[:,0]
    y = X_embedded[:,1]

    return x,y


def visualize_embeddings(text_to_embedding,reduce_fn="TSNE",write_to_file=False,file_name="embeddings"):
    items = list(sorted(text_to_embedding.items()))
    text =  [k for k,_ in items]
    normalize = lambda v: v/np.linalg.norm(v) if np.sum(v) != 0 else v
    vector_representation =  [normalize(v) for _,v in items]
    x,y = reduce_dims(vector_representation,type=reduce_fn)

    if write_to_file: 
        with open(file_name,"w",encoding="utf-8") as embedding_file:
            embedding_file.write("text\tx\ty\n")
            for a,x1,y1 in zip(text,x,y):
                embedding_file.write(a+"\t{}\t{}\n".format(x1,y1))


    plt.scatter(x,y)
    plt.show()

def check_embeddings_for_NAN(text_to_embedding):

    from sklearn.impute import SimpleImputer

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit([v for _,v in text_to_embedding.items()])
    for text,embedding in text_to_embedding.items():
        text_to_embedding[text] = imp.transform(np.reshape(embedding,(1,-1))).squeeze(0) #np.nan_to_num(embedding,nan=0.0)


def print_best_matches(text_to_embedding):
    for text,vector in sorted(list(text_to_embedding.items())):
        print(text)
        print("Best Match:")
        best_match = sorted(list(text_to_embedding.items()),key=lambda x: cosine_similarity(np.reshape(x[1],(1,-1)),np.reshape(vector,(1,-1))),reverse=True)
        print(best_match[1][0])
        print("\n")

def calculate_num_clusters(embeddings,kmax=30):
    sil = []
    embs = embeddings
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2, kmax+1):
      kmeans = KMeans(n_clusters = k).fit(embs)
      labels = kmeans.labels_
      sil.append((k,silhouette_score(embs, labels, metric = 'euclidean')))

    return max(sil,key=lambda x: x[1])

def run_kmeans(text_to_embedding,n_clusters):
    items = list(sorted(text_to_embedding.items()))
    text =  [k for k,_ in items]
    vector_representation =  [v/np.linalg.norm(v) for _,v in items]

    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(vector_representation)
    word_2_cluster = {}
    cluster_2_words = {}
    for i,l in enumerate(kmeans.labels_):
        word_2_cluster[" ".join(text[i])] = l
        if l not in cluster_2_words: cluster_2_words[l] = []
        cluster_2_words[l].append(" ".join(text[i]))


    for k,v in cluster_2_words.items():
        print(k)
        for paper in v:
            print(v)






