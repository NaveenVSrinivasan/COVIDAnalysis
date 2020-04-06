from argparse import ArgumentParser
import enum

from build_embedding_functions import *
from text_extraction_functions import *
from visualize_embedding_functions import *
import json


class Embedding(enum.Enum):
    SCIBERT = "SCIBERT"
    SCIBERT_TFIDF = "SCIBERT_TF_IDF"
    TFIDF = "TFIDF"
    MESH = "MESH"

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return Embedding[s]
        except KeyError:
            raise ValueError()


class Text(enum.Enum):
    ABSTRACTS = "ABSTRACTS"
    TITLES = "TITLES"
    BODY_TEXTS = "BODY_TEXTS"
    PARAGRAPHS = "PARAGRAPHS"

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return Text[s]
        except KeyError:
            raise ValueError()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--extraction_dir', '-extraction_dir',
                        help='extraction directory',
                        default='../../CORD-19-research-challenge/',
                        type=str)

    parser.add_argument('--use_text', '-use_text',
                        help='which text to extract to construct embeddings',
                        default=Text.ABSTRACTS,
                        type=Text.from_string,
                        choices=list(Text))

    parser.add_argument('--embedding_type', '-embedding_type',
                        help='which embedding to use',
                        default=Embedding.SCIBERT,
                        type=Embedding.from_string,
                        choices=list(Embedding))

    parser.add_argument('--load_file', '-load_file',
                        help='whether to load existing embeddings',
                        action='store_true')

    parser.add_argument('--write_file', '-write_file',
                        help='to write computed embeddings',
                        action='store_true')

    parser.add_argument('--embeddings_file', '-embeddings_file',
                        help='file to write embeddings to',
                        default='embeddings',
                        type=str)

    args = parser.parse_args()

    text_to_embeddings = None

    if args.load_file:
        print("-------Loading Embeddings File-------")
        with open(args.embeddings_file, "r") as json_file:
            text_to_embeddings = json.load(json_file)

    else:
        print("-------Retrieving Texts-------")
        texts = None
        if args.use_text == Text.ABSTRACTS:
            texts = extract_abstracts_precleaned(use_titles=True)
        elif args.use_text == Text.TITLES:
            texts = extract_titles(args.extraction_dir)

        print("-------Constructing Embeddings-------")
        text_to_embeddings = None
        if args.embedding_type == Embedding.SCIBERT:
            text_to_embeddings = build_scibert_embeds_paragraphs(texts)
        elif args.embedding_type == Embedding.SCIBERT_TFIDF:
            text_to_embeddings = build_scibert_embeds_tfidf_paragraphs(texts)
        elif args.embedding_type == Embedding.TFIDF:
            text_to_embeddings = build_tfidf_embeds_paragraphs(texts)
        elif args.embedding_type == Embedding.MESH:
            with open('../mesh/mesh_descriptors.txt','r') as mesh_file:
                features = set([x.strip() for x in mesh_file.readlines()])
            text_to_embeddings = build_scibert_embeds_mesh_paragraphs(texts,features)

        print("-------Writing Embeddings File-------")
        if args.write_file:
            for k in text_to_embeddings.keys():  # write embeddings to list so they can be stored as json
                text_to_embeddings[k] = text_to_embeddings[k].tolist()

            with open(args.embeddings_file, "w+") as json_file:
                json.dump(text_to_embeddings, json_file)

    print("-------Check Embeddings for NaNs-------")
    check_embeddings_for_NAN(text_to_embeddings) #remove any nans or infs in data

    # run_elbow(text_to_embeddings)
    # num_clusters,value = calculate_num_clusters([v for _,v in text_to_embeddings.items()],kmax=30) #calculate the number of clusters using siloutte score

    text, clusters = visualize_embeddings(text_to_embeddings, num_clusters=5, reduce_fn=Reduction.TSNE,
                                          write_to_file=True, file_name='embeddings_mesh') #visualize embeddings and write the reduced data to tsv file for visualizer
    #To visualize, change file in visualizer to this file location

    extract_cluster_names(text, clusters)
    # print_best_matches(text_to_embeddings) #for every abstract, print the best match
  
    # print(num_clusters,value)



