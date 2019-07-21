# script to generate generic features

import concurrent.futures
import os
import pickle
import sys
import traceback

import scispacy
import spacy
from nltk.translate.bleu_score import sentence_bleu
from nltk.metrics import edit_distance, jaccard_distance
from rouge import Rouge
from tqdm import tqdm

rouge = Rouge()
nlp = spacy.load('en_core_sci_md')

def extract_features(sent1, sent2, lowercase=False):
    ''' Main function to generate generic features
        Args (string): sent1, sent2 '''
        
    doc1 = nlp(sent1)
    doc2 = nlp(sent2)
    sent1 = [tok.text for tok in doc1]
    sent2 = [tok.text for tok in doc2]
    if lowercase:
        sent1 = [token.lower() for token in sent1]
        sent2 = [token.lower() for token in sent2]

    features = _generate_features(sent1, sent2, doc1, doc2)

    return features

def _ngram_overlap(sent1, sent2, n):
    if len(sent1) < n or len(sent2) < n:
        return [0.0, 0]

    sent1_ngrams = _extract_ngrams(sent1, n)
    sent2_ngrams = _extract_ngrams(sent2, n)

    overlap = 0
    for ngram in sent1_ngrams:
        if ngram in sent2_ngrams:
            overlap += min(sent1_ngrams[ngram], sent2_ngrams[ngram])

    return [overlap / (len(sent1) - n + 1), overlap]

def _extract_ngrams(sent, n):
    ngrams = {}
    for i in range(len(sent) - n + 1):
        ngram = tuple(sent[i:i+n])
        if ngram not in ngrams:
            ngrams[ngram] = 0
        ngrams[ngram] += 1
    return ngrams

def _pos_overlap(doc1, doc2, pos):
    sent1 = [tok.text for tok in doc1 if tok.pos_ == pos]
    sent2 = [tok.text for tok in doc2 if tok.pos_ == pos]
    overlaps = _ngram_overlap(sent1, sent2, 1)
    return overlaps

def _levenshtein(sent1, sent2, use_char):
    if use_char:
        sent1 = ' '.join(sent1)
        sent2 = ' '.join(sent2)
    abs_diff = edit_distance(sent1, sent2)
    return [abs_diff / len(sent1), abs_diff]

def _jaccard(sent1, sent2):
    sent1 = set(sent1)
    sent2 = set(sent2)
    return jaccard_distance(sent1, sent2)

def _rouge_score(sent1, sent2):
    if len(sent1) == 0 or len(sent2) == 0:
        return _zero_score()

    sent1 = ' '.join(sent1)
    sent2 = ' '.join(sent2)
    return rouge.get_scores(sent2, sent1)[0]

def _zero_score():
    return {'rouge-1': {'f': 0, 'p': 0, 'r': 0},
            'rouge-2': {'f': 0, 'p': 0, 'r': 0},
            'rouge-l': {'f': 0, 'p': 0, 'r': 0}}

def _bleu_score(sent1, sent2):
    return [sentence_bleu([sent1], sent2, weights=(1, 0, 0, 0)),
            sentence_bleu([sent1], sent2, weights=(0, 1, 0, 0)),
            sentence_bleu([sent1], sent2, weights=(0, 0, 1, 0)),
            sentence_bleu([sent1], sent2, weights=(0, 0, 0, 1))]

def _generate_features(sent1, sent2, doc1, doc2):
    unigram_overlap = _ngram_overlap(sent1, sent2, 1)
    bigram_overlap = _ngram_overlap(sent1, sent2, 2)
    trigram_overlap = _ngram_overlap(sent1, sent2, 3)
    pos_overlap = {pos: _pos_overlap(doc1, doc2, pos)
                   for pos in ['NOUN', 'VERB', 'ADJ', 'ADV']}
    leven_c = _levenshtein(sent1, sent2, True)
    leven_w = _levenshtein(sent1, sent2, False)
    jac = _jaccard(sent1, sent2)
    r_scores = _rouge_score(sent1, sent2)
    b_scores = _bleu_score(sent1, sent2)
    return [unigram_overlap[0], # unigram overlap (percent)
        bigram_overlap[0], # bigram overlap (percent)
        trigram_overlap[0], # trigram overlap (percent)
        unigram_overlap[1], # unigram overlap (absolute)
        bigram_overlap[1], # bigram overlap (absolute)
        trigram_overlap[1], # trigram overlap (absolute)

        pos_overlap['NOUN'][0], # noun overlap (percent)
        pos_overlap['NOUN'][1],  # noun overlap (absolute)
        pos_overlap['VERB'][0],  # verb overlap (percent)
        pos_overlap['VERB'][1],  # verb overlap (absolute)
        pos_overlap['ADJ'][0],  # adj overlap (percent)
        pos_overlap['ADJ'][1],  # adj overlap (absolute)
        pos_overlap['ADV'][0],  # adv overlap (percent)
        pos_overlap['ADV'][1],  # adv overlap (absolute)
    
        leven_c[0], # character based edit distance (percent)
        leven_c[1], # character based edit distance (absolute)
        leven_w[0], # token based edit distance (percent)
        leven_w[1], # token based edit distance (absolute)
        jac, # jaccard distance
        r_scores['rouge-1']['f'], # ROUGE-1
        r_scores['rouge-2']['f'], # ROUGE-2
        r_scores['rouge-l']['f'], # ROUGE-L
        b_scores[0], # BLEU-1
        b_scores[1], # BLEU-2
        b_scores[2], # BLEU-3
        b_scores[3], # BLEU-4
        len(sent1) - len(sent2), # length difference (signed)
    ]

if __name__ == '__main__':
    for namespace in ('train', 'dev'):
        file_path = f'data/{namespace}.tsv'
        dump_path = f'data/mt_dnn/{namespace}_generic_features.pkl'

        print(f'processing {namespace}.tsv...')

        with open(file_path, 'r') as tsvfile:
            tsv_lines = tsvfile.readlines()

        features_list = []
        for row in tqdm(tsv_lines):
            columns = row.split('\t')
            sent1 = columns[1]
            sent2 = columns[2]
            features_list.append(extract_features(sent1, sent2))

        with open(dump_path, 'wb') as dumpfile:
            pickle.dump(features_list, dumpfile)
