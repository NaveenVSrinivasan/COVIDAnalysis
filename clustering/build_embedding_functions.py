import numpy as np
import torch
from tqdm import tqdm
from transformers import *
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_embeds_paragraphs(paragraphs):
    """
    :param paragraphs: list of lists of sentences
    :return: dictionary paragraphs to tfidf embeddings
    """
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    vectorizer = TfidfVectorizer()

    print("Processing {} papers".format(len(paragraphs)))

    corpus = [" ".join(tokenizer.tokenize(s)) for s in a]
    X = vectorizer.fit_transform(corpus)
    text_to_embeddings = {" ".join(a): np.asarray(v).flatten() for a, v in zip(paragraphs, X.todense())}

    return text_to_embeddings


def build_scibert_embeds_sentences(sentences):
    """
    :param sentences: list of sentences
    :return: dictionary of sentences to sciBERT embeddings
    """
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

    print("Processing {} sentences".format(len(sentences)))

    ######Build Vectors######################
    sentences_to_embeddings = {}

    for sentence in tqdm(sentences):
        tokenized = torch.tensor([tokenizer.encode(sentence)])  # .cuda()
        hidden_states, cls_emb = model(tokenized)
        sentences_to_embeddings[sentence] = torch.sum(hidden_states, 1).view(-1).data.numpy()

    return sentences_to_embeddings


def build_scibert_embeds_paragraphs(paragraphs):
    """
    :param paragraphs: list of lists of sentences
    :return: dictionary paragraphs to sciBERT embeddings
    """
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    
    print("Processing {} papers".format(len(paragraphs)))

    ######Build Vectors######################
    vector_representation = []
    for a in tqdm(paragraphs):
        all_sentence_embeds = []
        # print(a)
        for sentence in a:
            tokenized = torch.tensor([tokenizer.encode(sentence)]) #.cuda()
            hidden_states,cls_emb = model(tokenized)
            all_sentence_embeds.append(torch.sum(hidden_states,1).view(-1))
        abstract_embeddings = torch.stack(all_sentence_embeds)
        vector_representation.append(torch.sum(abstract_embeddings,axis=0).data.numpy())

    text_to_embeddings = {" ".join(a): v for a, v in zip(paragraphs, vector_representation)}

    return text_to_embeddings


def build_scibert_embeds_documents_avg(documents):
    """
    :param documents: list of lists of lists of sentences
    :return: dictionaries representing sentences to embeddings, paragraphs to embeddings, and documents to embeddings
    """
    sentences = [sentence for document in documents for paragraph in document for sentence in paragraph]
    sentences_to_embeddings = build_scibert_embeds_sentences(sentences)
    documents_to_embeddings = {}
    paragraphs_to_embeddings = {}
    for document in documents:
        all_paragraph_embeds = []
        for paragraph in document:
            all_sentence_embeds = []
            for sentence in paragraph:
                all_sentence_embeds.append(torch.tensor(sentences_to_embeddings[sentence]))
            paragraph_embedding = torch.mean(torch.stack(all_sentence_embeds), axis=0)
            all_paragraph_embeds.append(paragraph_embedding)
            paragraphs_to_embeddings[" ".join(paragraph)] = paragraph_embedding.data.numpy()
        document_embedding = torch.mean(torch.stack(all_paragraph_embeds), axis=0)
        paragraph_texts = [" ".join(paragraph) for paragraph in document]
        documents_to_embeddings[" ".join(paragraph_texts)] = document_embedding.data.numpy()
    return documents_to_embeddings, paragraphs_to_embeddings, sentences_to_embeddings


def build_scibert_embeds_tfidf_paragraphs(paragraphs):
    """
    :param paragraphs: list of lists of sentences
    :return: dictionary paragraphs to sciBERT embeddings with tfidf weights
    """
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    
    print("Processing {} papers".format(len(paragraphs)))

    vectorizer = TfidfVectorizer()

    corpus = [" ".join([" ".join(tokenizer.tokenize(s)) for s in a]) for a in paragraphs]
    all_sentence_embeds = []
    X = vectorizer.fit_transform(corpus)
    words_to_index = {w:i for i,w in enumerate(vectorizer.get_feature_names())}  
    with tqdm(total=len(paragraphs)) as pbar:
        for tfidf_score,text in tqdm(zip(X,paragraphs)):
            tfidf_score = tfidf_score.toarray()
            for sentence in text:
                tokenized = tokenizer.tokenize(sentence)
                weights = []
                for w in tokenized:
                    if w in words_to_index:
                        weights.append(tfidf_score[0][words_to_index[w]])
                    else:
                        weights.append(0.0)

                weights = torch.tensor(weights)
                weights = weights / torch.sum(weights) if torch.sum(weights) > 0 else weights

                # print(weights)

                hidden_states,cls_emb = model(torch.tensor([tokenizer.encode(sentence,add_special_tokens=False)]))
                # print(hidden_states.size())
                # print(weights.size())

                new_hidden_states = torch.mm(hidden_states.permute((0,2,1)).squeeze(0),weights.reshape(-1,1))
                all_sentence_embeds.append(new_hidden_states.view(-1).data.numpy())
            pbar.update(1)

    text_to_embeddings = {" ".join(a):v for a,v in zip(paragraphs, all_sentence_embeds)}

    return text_to_embeddings



def build_scibert_embeds_mesh_paragraphs(paragraphs,text_features):
    """
    :param paragraphs: list of lists of sentences
    :return: dictionary paragraphs to sciBERT embeddings with tfidf weights
    """
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    
    print("Processing {} papers".format(len(paragraphs)))

    # vectorizer = TfidfVectorizer()

    corpus = [" ".join([" ".join(tokenizer.tokenize(s)) for s in a]) for a in paragraphs]
    all_sentence_embeds = []
    # X = vectorizer.fit_transform(corpus)
    # words_to_index = {w:i for i,w in enumerate(vectorizer.get_feature_names())}  

    with tqdm(total=len(paragraphs)) as pbar:
        for text in tqdm(paragraphs[:100]):
            for sentence in text:
                tokenized = tokenizer.tokenize(sentence)
                weights = []
                for w in tokenized:
                    # print(w)
                    if w in text_features:
                        # print("\tFOUND")
                        weights.append(1.0) #tfidf_score[0][words_to_index[w]])
                    else:
                        weights.append(0.0)

                weights = torch.tensor(weights)
                # weights = weights / torch.sum(weights) if torch.sum(weights) > 0 else weights

                # print(weights)

                hidden_states,cls_emb = model(torch.tensor([tokenizer.encode(sentence,add_special_tokens=False)]))
                # print(hidden_states.size())
                # print(weights.size())

                new_hidden_states = torch.mm(hidden_states.permute((0,2,1)).squeeze(0),weights.reshape(-1,1))
                all_sentence_embeds.append(new_hidden_states.view(-1).data.numpy())
            pbar.update(1)

    text_to_embeddings = {" ".join(a):v for a,v in zip(paragraphs, all_sentence_embeds)}

    return text_to_embeddings
