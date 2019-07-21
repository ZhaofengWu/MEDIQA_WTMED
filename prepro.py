# Copyright (c) Microsoft. All rights reserved.
import os
import json
import tqdm
import pickle
import re
import collections
import argparse
from sys import path
from data_utils.vocab import Vocabulary
from pytorch_pretrained_bert.tokenization import BertTokenizer, BasicTokenizer, WordpieceTokenizer, load_vocab
from data_utils.log_wrapper import create_logger
from data_utils.label_map import GLOBAL_MAP
from data_utils.glue_utils import *
DEBUG_MODE=False
MAX_SEQ_LEN = 512

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
logger = create_logger(__name__, to_disk=True, log_file='bert_data_proc_512.log')

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length.
    Copyed from https://github.com/huggingface/pytorch-pretrained-BERT
    """
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def build_data(data, dump_path, max_seq_len=MAX_SEQ_LEN, is_train=True, tolower=True):
    """Build data of sentence pair tasks
    """
    with open(dump_path, 'w', encoding='utf-8') as writer:
        for idx, sample in enumerate(data):
            ids = sample['uid']
            premise = bert_tokenizer.tokenize(sample['premise'])
            hypothesis = bert_tokenizer.tokenize(sample['hypothesis'])
            label = sample['label']
            _truncate_seq_pair(premise, hypothesis, max_seq_len - 3)
            input_ids =bert_tokenizer.convert_tokens_to_ids(['[CLS]'] + hypothesis + ['[SEP]'] + premise + ['[SEP]'])
            type_ids = [0] * ( len(hypothesis) + 2) + [1] * (len(premise) + 1)
            features = {'uid': ids, 'label': label, 'token_id': input_ids, 'type_id': type_ids}
            for k, v in sample.items():
                if k in {'parse_id_a', 'parse_id_b'}:
                    features[k] = v
            writer.write('{}\n'.format(json.dumps(features, separators=(',',':'))))

def build_qnli(data, dump_path, max_seq_len=MAX_SEQ_LEN, is_train=True, tolower=True, gold_path=None):
    """Build QNLI as a pair-wise ranking task
    """
    with open(dump_path, 'w', encoding='utf-8') as writer:
        for idx, sample in enumerate(data):
            ids = sample['uid']
            premise = bert_tokenizer.tokenize(sample['premise'])
            hypothesis_1 = bert_tokenizer.tokenize(sample['hypothesis'][0])
            hypothesis_2 = bert_tokenizer.tokenize(sample['hypothesis'][1])
            label = sample['label']
            _truncate_seq_pair(premise, hypothesis_1, max_seq_len - 3)
            input_ids_1 =bert_tokenizer.convert_tokens_to_ids(['[CLS]'] + hypothesis_1 + ['[SEP]'] + premise + ['[SEP]'])
            type_ids_1 = [0] * ( len(hypothesis_1) + 2) + [1] * (len(premise) + 1)
            _truncate_seq_pair(premise, hypothesis_2, max_seq_len - 3)
            input_ids_2 =bert_tokenizer.convert_tokens_to_ids(['[CLS]'] + hypothesis_2 + ['[SEP]'] + premise + ['[SEP]'])
            type_ids_2 = [0] * ( len(hypothesis_2) + 2) + [1] * (len(premise) + 1)
            features = {'uid': ids, 'label': label, 'token_id': [input_ids_1, input_ids_2], 'type_id': [type_ids_1, type_ids_2], 'ruid':sample['ruid'], 'olabel':sample['olabel']}
            writer.write('{}\n'.format(json.dumps(features)))

def build_data_single(data, dump_path, max_seq_len=MAX_SEQ_LEN):
    """Build data of single sentence tasks
    """
    with open(dump_path, 'w', encoding='utf-8') as writer:
        for idx, sample in enumerate(data):
            ids = sample['uid']
            premise = bert_tokenizer.tokenize(sample['premise'])
            label = sample['label']
            if len(premise) >  max_seq_len - 3:
                premise = premise[:max_seq_len - 3] 
            input_ids =bert_tokenizer.convert_tokens_to_ids(['[CLS]'] + premise + ['[SEP]'])
            type_ids = [0] * ( len(premise) + 2)
            features = {'uid': ids, 'label': label, 'token_id': input_ids, 'type_id': type_ids}
            writer.write('{}\n'.format(json.dumps(features)))

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing GLUE/SNLI/SciTail dataset.')
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--root_dir', type=str, default='data')
    args = parser.parse_args()
    return args

def main(args):
    root = args.root_dir
    assert os.path.exists(root)

    mednli_train_path = os.path.join(root, 'train.tsv')
    mednli_dev_path = os.path.join(root, 'dev.tsv')
    
    mednli_train_data = load_mednli(mednli_train_path, GLOBAL_MAP['mednli'])
    mednli_dev_data = load_mednli(mednli_dev_path, GLOBAL_MAP['mednli'])
    logger.info('Loaded {} mednli train samples'.format(len(mednli_train_data)))
    logger.info('Loaded {} mednli dev samples'.format(len(mednli_dev_data)))

    mt_dnn_root = os.path.join(root, 'mt_dnn')
    if not os.path.isdir(mt_dnn_root):
        os.mkdir(mt_dnn_root)

    mednli_train_fout = os.path.join(mt_dnn_root, 'mednli_train.json')
    mednli_dev_fout = os.path.join(mt_dnn_root, 'mednli_dev.json')
    build_data(mednli_train_data, mednli_train_fout)
    build_data(mednli_dev_data, mednli_dev_fout)
    logger.info('done with mednli')

if __name__ == '__main__':
    args = parse_args()
    main(args)
