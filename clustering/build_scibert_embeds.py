from argparse import ArgumentParser

from build_embedding_functions import *
from text_extraction_functions import *
from visualize_embedding_functions import *
import json


class Embedding(Enum):
    SCIBERT = "SCIBERT"
    SCIBERT_TFIDF = "SCIBERT_TF_IDF"
    TFIDF = "TFIDF"


class Text(Enum):
    ABSTRACTS = "Abstracts"
    TITLES = "Titles"
    BODY_TEXTS = "Body_texts"
    PARAGRAPHS = "Paragraphs"


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--extraction_dir', '-extraction_dir',
                        help='extraction directory',
                        default='../../CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv',
                        type=str)

    parser.add_argument('--use_text', '-use_text',
                        help='which text to extract to construct embeddings',
                        default=Text.ABSTRACTS,
                        type=Text)

    parser.add_argument('--embedding_type', '-embedding_type',
                        help='which embedding to use',
                        default=Embedding.SCIBERT_TFIDF,
                        type=Embedding)

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

    # Load text to embeddings directly from embeddings file
    if args.load_file:
        with open(args.embeddings_file, "r") as json_file:
            text_to_embeddings = json.load(json_file)

    else:
        # Retrieve texts
        texts = None
        if args.use_text == Text.ABSTRACTS:
            texts = extract_abstracts(args.extraction_dir, remove_ints=True)
        elif args.use_text == Text.TITLES:
            texts = extract_titles(args.extraction_dir)

        # Construct embeddings
        text_to_embeddings = None
        if args.embedding_type == Embedding.SCIBERT:
            text_to_embeddings = build_scibert_embeds(texts)
        elif args.embedding_type == Embedding.SCIBERT_TFIDF:
            text_to_embeddings = build_scibert_embeds_tf_idf(texts)
        elif args.embedding_type == Embedding.TFIDF:
            text_to_embeddings = build_tfidf_embeds(texts)

        # Write file
        if args.write_file:
            for k in text_to_embeddings.keys():  # write embeddings to list so they can be stored as json
                text_to_embeddings[k] = text_to_embeddings[k].tolist()

            with open(args.embeddings_file, "w+") as json_file:
                json.dump(text_to_embeddings, json_file)

    check_embeddings_for_NAN(text_to_embeddings) #remove any nans or infs in data

    # run_elbow(text_to_embeddings)
    # num_clusters,value = calculate_num_clusters([v for _,v in text_to_embeddings.items()],kmax=30) #calculate the number of clusters using siloutte score

    text, clusters = visualize_embeddings(text_to_embeddings, num_clusters=15, reduce_fn=Reduction.TSNE,
                                          write_to_file=False, file_name='embeddings_all') #visualize embeddings and write the reduced data to tsv file for visualizer
    #To visualize, change file in visualizer to this file location

    extract_cluster_names(text,clusters)
    print_best_matches(text_to_embeddings) #for every abstract, print the best match

  
    # print(num_clusters,value)



