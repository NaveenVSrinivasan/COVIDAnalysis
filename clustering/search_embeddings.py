from argparse import ArgumentParser
import enum
from create_embeddings import Embedding
from visualize_embedding_functions import *
import json

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--embedding_type', '-embedding_type',
                        help='which embedding to use',
                        default=Embedding.SCIBERT,
                        type=Embedding.from_string,
                        choices=list(Embedding))

    parser.add_argument('--embeddings_file', '-embeddings_file',
                        help='file to write embeddings to',
                        default='embeddings',
                        type=str)

    parser.add_argument('--num_top_abstracts', '-num_top_abstracts',
                        help='number of top abstracts to return',
                        default=5,
                        type=int)

    args = parser.parse_args()

    var = input("Please enter a statement or question: ")

    print("-------Loading Embeddings File-------")
    with open(args.embeddings_file, "r") as json_file:
        text_to_embeddings = json.load(json_file)

    print("-------Constructing Question Embedding-------")
    var_embedding = None
    if args.embedding_type == Embedding.SCIBERT:
        var_embedding = build_scibert_embeds_paragraphs([[var]])
    elif args.embedding_type == Embedding.SCIBERT_TFIDF:
        var_embedding = build_scibert_embeds_tfidf_paragraphs([[var]])
    elif args.embedding_type == Embedding.TFIDF:
        var_embedding = build_tfidf_embeds_paragraphs([[var]])
    elif args.embedding_type == Embedding.MESH:
        with open('../mesh/mesh_descriptors.txt', 'r') as mesh_file:
            features = set([x.strip() for x in mesh_file.readlines()])
        var_embedding = build_scibert_embeds_mesh_paragraphs([[var]], features)

    var_embedding = list(var_embedding.values())[0]

    print(search_top(var_embedding, text_to_embeddings, args.num_top_abstracts))

