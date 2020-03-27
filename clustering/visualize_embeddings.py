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
from transformers import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from numpy import inf
from nltk.corpus import stopwords



def reduce_dims(embeddings,type="TSNE"):
    if type == "TSNE":
        X_embedded = TSNE(n_components=2).fit_transform(embeddings)
    elif type == "PCA":
        X_embedded = PCA(n_components=2).fit_transform(embeddings)

    x = X_embedded[:,0]
    y = X_embedded[:,1]

    return x,y


def visualize_embeddings(text_to_embedding,reduce_fn="TSNE",num_clusters,write_to_file=False,file_name="embeddings"):
    items = list(sorted(text_to_embedding.items()))
    text =  [k for k,_ in items]
    normalize = lambda v: v/np.linalg.norm(v) if np.sum(v) != 0 else v
    vector_representation =  [normalize(v) for _,v in items]
    x,y = reduce_dims(vector_representation,type=reduce_fn)

    labels = run_kmeans(vector_representation,num_clusters)
    # print(num)

    if write_to_file: 
        with open(file_name,"w",encoding="utf-8") as embedding_file:
            embedding_file.write("text\tx\ty\tcluster\n")
            for a,x1,y1,cluster in zip(text,x,y,labels):
                embedding_file.write(a+"\t{}\t{}\t{}\n".format(x1,y1,cluster))


    plt.scatter(x,y)
    plt.show()

    return text,labels

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

    print(sil)
    return max(sil,key=lambda x: x[1])

def run_kmeans(embeddings,n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    return kmeans.labels_
    
def run_elbow(text_to_embedding):
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
    from yellowbrick.cluster import KElbowVisualizer

    items = list(sorted(text_to_embedding.items()))
    text =  [k for k,_ in items]
    normalize = lambda v: v/np.linalg.norm(v) if np.sum(v) != 0 else v
    vector_representation =  [normalize(v) for _,v in items]

    # Generate synthetic dataset with 8 random clusters

    # Instantiate the clustering model and visualizer
    model = KMeans()
    visualizer = KElbowVisualizer(
        model, k=(2,35), metric="silhouette", timings=False,locate_elbow=True
    )

    visualizer.fit(vector_representation)        # Fit the data to the visualizer
    visualizer.show() 


def extract_cluster_names(text,labels):
    label_to_all_text = {l:"" for l in labels}
    for t,l in zip(text,labels):
        label_to_all_text[l]+= " "+t

    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    corpus = [t for _,t in sorted(label_to_all_text.items(),key=lambda x: x[0])]
    labels = [label for label,_ in sorted(label_to_all_text.items(),key=lambda x: x[0])]
    X = vectorizer.fit_transform(corpus)

    words_to_index = {w:i for i,w in enumerate(vectorizer.get_feature_names())}  
    all_words = vectorizer.get_feature_names()

    for tfidf_score,l in tqdm(zip(X,labels)):
            tfidf_score = np.reshape(tfidf_score.toarray(),(-1))
            most_characteristic_words = sorted(all_words,key=lambda x: tfidf_score[words_to_index[x]],reverse=True)[:10]
            print(most_characteristic_words)

