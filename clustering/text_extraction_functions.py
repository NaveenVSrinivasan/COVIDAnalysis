import glob
import json
import nltk.data
import re
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

    print("ABSTRACTS TOTAL:",len(abstracts))
    return abstracts

def extract_abstracts_precleaned(use_titles=True):
    """
    :param raw_abstracts: dict from title to abtract
    :param use_titles: if titles included in abstract pull
    :return: list of abstracts split by sentence
    """
    with open("title_to_abstract_mapping.json") as json_file:
        raw_abstracts = json.load(json_file)

    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    abstracts = []

    for title, abstract in raw_abstracts.items():
        if use_titles:
            abstract = [title] + abstract

        sentences = [sent_detector.tokenize(sentence) for sentence in abstract if len(sentence.split()) > 0]

        if len(sentences) > 0:
            abstracts.append(sentences)

    print("ABSTRACTS TOTAL:",len(abstracts))
    return abstracts


def extract_documents(directory, remove_ints=False, use_abstracts=True):
    """
    :param directory: directory to recursively find json files
    :param remove_ints: optional argument to remove digits and websites
    :param use_abstracts: if abstracts (and titles) included
    :return: list of documents which are lists of paragraphs which are lists of sentences
    """
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    documents = []

    for file in sorted(list(glob.glob(directory+'**/*.json', recursive=True))):
        paper_data = json.load(open(file))               # <-- Things done for each file
        paragraphs = []
        if use_abstracts:
            if len(paper_data['abstract']) > 0:
                all_text = " ".join([x["text"] for x in paper_data['abstract']])
                all_text = paper_data['metadata']['title']+". " + all_text
                all_text = all_text.replace("All rights reserved. No reuse allowed without permission.","")
                all_text = all_text.replace("Abstract", "")

                sentences = [s for s in sent_detector.tokenize(all_text.strip()) if "author/funder." not in s and len(s.split()) > 1]

                if remove_ints:
                    clean = lambda x: " ".join([word for word in re.split("\W+", x) if not word.isdigit() and "http" not in word])
                    sentences = [clean(s) for s in sentences]

                if len(sentences) > 0:
                    paragraphs.append(sentences)

        for paragraph in paper_data['body_text']:
            all_text = paragraph['text']

            all_text = all_text.replace("All rights reserved. No reuse allowed without permission.", "")
            all_text = all_text.replace("author/funder", "")

            sentences = [s for s in sent_detector.tokenize(all_text.strip()) if 'copyright' not in s and
                         'medRxiv' not in s and len(s.split()) > 1]

            if remove_ints:
                clean = lambda x: " ".join([word for word in re.split("\W+", x) if not word.isdigit() and "http" not in word])
                sentences = [clean(s) for s in sentences]

            if len(sentences) > 0:
                paragraphs.append(sentences)

        documents.append(paragraphs)

    return documents
