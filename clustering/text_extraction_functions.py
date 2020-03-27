import glob
import json
import nltk.data
from transformers import *


def extract_titles(directory):
    """
    :param directory: directory to recursively find json files
    :return: list of abstracts split by sentence
    """
    titles = []

    for file in sorted(list(glob.glob(directory+'**/*.json', recursive=True))):
        paper_data = json.load(open(file))               # <-- Things done for each file
        title = paper_data['metadata']['title']
        if len(title.split()) > 0:
            titles.append([title])
    return titles


def extract_abstracts(directory, remove_ints=False, use_titles=True):
    """
    :param directory: directory to recursively find json files
    :param remove_ints: optional argument to remove digits and websites
    :param use_titles: if titles included in abstract pull
    :return: list of abstracts split by sentence
    """
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    abstracts = []

    for file in sorted(list(glob.glob(directory+'**/*.json', recursive=True))):
        paper_data = json.load(open(file))               # <-- Things done for each file
        if len(paper_data['abstract']) > 0:
            all_text = " ".join([x["text"] for x in paper_data['abstract']])
            if use_titles:
                all_text = paper_data['metadata']['title']+". " + all_text
            all_text = all_text.replace("All rights reserved. No reuse allowed without permission.","")
            if "author/funder." in all_text:
                all_text = all_text[all_text.find("author/funder.")+len("author/funder."):]
            sentences = [s for s in sent_detector.tokenize(all_text.strip()) if "author/funder." not in s and len(s.split()) > 1]

            if remove_ints:
                clean = lambda x: " ".join([word for word in x.split() if not word.isdigit() and "http" not in word])
                sentences = [clean(s) for s in sentences]
                sentences = [s for s in sentences if len(s.split()) > 1]

            if len(sentences) > 0:
                abstracts.append(sentences)
    return abstracts


def extract_body_text(directory, remove_ints=False, use_abstracts=True):
    pass


def extract_all_paragraphs(directory, remove_ints=False):
    pass
