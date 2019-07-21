# script to generate domain features

import csv
import pickle

import scispacy
import spacy
from tqdm import tqdm

bionlp = spacy.load('en_ner_bionlp13cg_md')
# entity types of 'en_ner_bionlp13cg_md'
bionlp_entities = ['CANCER', 'ORGAN', 'TISSUE', 'ORGANISM', 'CELL', 'AMINO_ACID', 'GENE_OR_GENE_PRODUCT', 'SIMPLE_CHEMICAL', 'ANATOMICAL_SYSTEM',
               'IMMATERIAL_ANATOMICAL_ENTITY', 'MULTI-TISSUE_STRUCTURE', 'DEVELOPING_ANATOMICAL_STRUCTURE', 'ORGANISM_SUBDIVISION', 'CELLULAR_COMPONENT',
               'ORGANISM_SUBSTANCE', 'PATHOLOGICAL_FORMATION']
bionlp_dict = {ent: idx for idx, ent in enumerate(bionlp_entities)}

bc5cdr = spacy.load('en_ner_bc5cdr_md')
# entity types of 'en_ner_bc5cdr_md'
bc5cdr_entities = ['DISEASE', 'CHEMICAL']
bc5cdr_dict = {ent: idx for idx, ent in enumerate(bc5cdr_entities)}

def extract_features(sent1, sent2):
    ''' Main function to generate domain features
        Args (string): sent1, sent2 '''

    bionlp_doc1 = bionlp(sent1)
    bionlp_doc2 = bionlp(sent2)
    bc5cdr_doc1 = bc5cdr(sent1)
    bc5cdr_doc2 = bc5cdr(sent2)
    features = _generate_features(bionlp_doc1, bionlp_doc2, bc5cdr_doc1, bc5cdr_doc2)
    return features

def _one_hot_entities(bionlp_doc1, bionlp_doc2, bc5cdr_doc1, bc5cdr_doc2):
    bionlp_sent1_ent = [0] * len(bionlp_dict)
    bionlp_sent2_ent = [0] * len(bionlp_dict)
    bc5cdr_sent1_ent = [0] * len(bc5cdr_dict)
    bc5cdr_sent2_ent = [0] * len(bc5cdr_dict)

    for ent in bionlp_doc1.ents:
        bionlp_sent1_ent[bionlp_dict[ent.label_]] += 1
    for ent in bionlp_doc2.ents:
        bionlp_sent2_ent[bionlp_dict[ent.label_]] += 1
    for ent in bc5cdr_doc1.ents:
        bc5cdr_sent1_ent[bc5cdr_dict[ent.label_]] += 1
    for ent in bc5cdr_doc2.ents:
        bc5cdr_sent2_ent[bc5cdr_dict[ent.label_]] += 1
    return bionlp_sent1_ent + bc5cdr_sent1_ent + bionlp_sent2_ent + bc5cdr_sent2_ent

def _entity_overlap(bionlp_doc1, bionlp_doc2, bc5cdr_doc1, bc5cdr_doc2):
    bionlp_t_set1 = {ent.text for ent in bionlp_doc1.ents}
    bionlp_t_set2 = {ent.text for ent in bionlp_doc2.ents}
    bionlp_t_intersect = bionlp_t_set1.intersection(bionlp_t_set2)
    bc5cdr_t_set1 = {ent.text for ent in bc5cdr_doc1.ents}
    bc5cdr_t_set2 = {ent.text for ent in bc5cdr_doc2.ents}
    bc5cdr_t_intersect = bc5cdr_t_set1.intersection(bc5cdr_t_set2)
    ent_t_overlap = len(bionlp_t_intersect) + len(bc5cdr_t_intersect)  # entity t overlap

    bionlp_e_set1 = {ent.label_ for ent in bionlp_doc1.ents}
    bionlp_e_set2 = {ent.label_ for ent in bionlp_doc2.ents}
    bionlp_e_intersect = bionlp_e_set1.intersection(bionlp_e_set2)
    bc5cdr_e_set1 = {ent.label_ for ent in bc5cdr_doc1.ents}
    bc5cdr_e_set2 = {ent.label_ for ent in bc5cdr_doc2.ents}
    bc5cdr_e_intersect = bc5cdr_e_set1.intersection(bc5cdr_e_set2)
    ent_e_overlap = len(bionlp_e_intersect) + len(bc5cdr_e_intersect)  # entity e overlap

    return ent_t_overlap, ent_e_overlap

def _generate_features(bionlp_doc1, bionlp_doc2, bc5cdr_doc1, bc5cdr_doc2):
    one_hot = _one_hot_entities(bionlp_doc1, bionlp_doc2, bc5cdr_doc1, bc5cdr_doc2)
    ent_t_overlap, ent_e_overlap = _entity_overlap(bionlp_doc1, bionlp_doc2, bc5cdr_doc1, bc5cdr_doc2)

    features = one_hot
    features.append(ent_t_overlap)
    features.append(ent_e_overlap)
    return features

if __name__ == '__main__':
    for namespace in ('train', 'dev'):
        file_path = f'data/{namespace}.tsv'
        dump_path = f'data/mt_dnn/{namespace}_domain_features.pkl'

        print(f'processing {namespace}.tsv...')

        with open(file_path, 'r') as tsvfile:
            tsv_lines = tsvfile.readlines()

        features_list = []
        for row in tqdm(tsv_lines):
            columns = row.split('\t')
            sent1 = columns[1]
            sent2 = columns[2]
            features = extract_features(sent1, sent2)
            features_list.append(features)

        with open(dump_path, 'wb') as dumpfile:
            pickle.dump(features_list, dumpfile)
