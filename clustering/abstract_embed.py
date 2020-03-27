import os                         
import glob       
import json                                                          
import torch
import numpy as np
import nltk.data
from tqdm import tqdm
from transformers import *
from sklearn.feature_extraction.text import TfidfVectorizer


#######Load Abstract Data/Clean Text#################
def extract_abstracts(directory):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    abstracts = []

    for file in sorted(list(glob.glob(directory+'/*.json'))):
        paper_data = json.load(open(file))               # <-- Things done for each file
        if len(paper_data['abstract']) > 0: 
            all_text = " ".join([x["text"] for x in paper_data['abstract']])
            all_text = all_text.replace("All rights reserved. No reuse allowed without permission.","")
            if "author/funder." in all_text:
                all_text = all_text[all_text.find("author/funder.")+len("author/funder."):]
            sentences = [s for s in sent_detector.tokenize(all_text.strip()) if "author/funder." not in s and len(s.split()) > 1]
            if len(sentences) > 0: abstracts.append(sentences)
    return abstracts

#Takes list of list of sentences
def build_scibert_embeds(abstracts):
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    
    print("Processing {} papers".format(len(abstracts)))

    ######Build Vectors######################
    vector_representation = []
    for a in tqdm(abstracts):
        all_sentence_embeds = []
        # print(a)
        for sentence in a:
            tokenized = torch.tensor([tokenizer.encode(sentence)]) #.cuda()
            hidden_states,cls_emb = model(tokenized)
            all_sentence_embeds.append(torch.sum(hidden_states,1).view(-1))
        abstract_embeddings = torch.stack(all_sentence_embeds)
        vector_representation.append(torch.sum(abstract_embeddings,axis=0).data.numpy())


    text_to_embeddings = {" ".join(a):v for a,v in zip(abstracts,vector_representation)}


    return text_to_embeddings


def build_scibert_embeds_tf_idf(abstracts):
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    
    print("Processing {} papers".format(len(abstracts)))


    vectorizer = TfidfVectorizer()

    corpus = [" ".join([" ".join(tokenizer.tokenize(s)) for s in a]) for a in abstracts]
    all_sentence_embeds = []
    X = vectorizer.fit_transform(corpus)
    words_to_index = {w:i for i,w in enumerate(vectorizer.get_feature_names())}  
    with tqdm(total=len(abstracts)) as pbar:
        for tfidf_score,text in tqdm(zip(X,abstracts)):
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

                print(weights)

                hidden_states,cls_emb = model(torch.tensor([tokenizer.encode(sentence,add_special_tokens=False)]))
                # print(hidden_states.size())
                # print(weights.size())

                new_hidden_states = torch.mm(hidden_states.permute((0,2,1)).squeeze(0),weights.reshape(-1,1))
                all_sentence_embeds.append(new_hidden_states.view(-1).data.numpy())
            pbar.update(1)

    text_to_embeddings = {" ".join(a):v for a,v in zip(abstracts,all_sentence_embeds)}

    return text_to_embeddings