# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import os
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from module.dropout_wrapper import DropoutWrapper
from pytorch_pretrained_bert.modeling import BertConfig, BertEncoder, BertLayerNorm, BertModel
from module.san import SANClassifier, Classifier

from .tree import Tree
from .treelstm import BinaryTreeLSTM

class SANBertNetwork(nn.Module):
    def __init__(self, opt, bert_config=None,
                 use_parse=False, embedding_matrix=None, token2idx=None, stx_parse_dim=None, unked_words=None,
                 use_generic_features=False, num_generic_features=None, use_domain_features=False, num_domain_features=None, feature_dim=None):
        super(SANBertNetwork, self).__init__()
        self.dropout_list = []
        self.bert_config = BertConfig.from_dict(opt)
        self.bert = BertModel(self.bert_config)
        if opt['update_bert_opt'] > 0:
            for p in self.bert.parameters():
                p.requires_grad = False
        mem_size = self.bert_config.hidden_size
        self.scoring_list = nn.ModuleList()
        labels = [int(ls) for ls in opt['label_size'].split(',')]
        task_dropout_p = opt['tasks_dropout_p']
        self.bert_pooler = None

        self.use_parse = use_parse
        self.stx_parse_dim = stx_parse_dim
        self.use_generic_features = use_generic_features
        self.use_domain_features = use_domain_features

        clf_dim = self.bert_config.hidden_size
        if self.use_parse:
            self.treelstm = BinaryTreeLSTM(self.stx_parse_dim, embedding_matrix.clone(), token2idx, unked_words=unked_words)
            parse_clf_dim = self.stx_parse_dim * 2
            clf_dim += parse_clf_dim
            self.parse_clf = nn.Linear(parse_clf_dim, labels[0])
        if self.use_generic_features:
            self.generic_feature_proj = nn.Linear(num_generic_features, num_generic_features * feature_dim)
            generic_feature_clf_dim = num_generic_features * feature_dim
            clf_dim += generic_feature_clf_dim
            self.generic_feature_clf = nn.Linear(generic_feature_clf_dim, labels[0])
        if self.use_domain_features:
            self.domain_feature_proj = nn.Linear(num_domain_features, num_domain_features * feature_dim)
            domain_feature_clf_dim = num_domain_features * feature_dim
            clf_dim += domain_feature_clf_dim
            self.domain_feature_clf = nn.Linear(domain_feature_clf_dim, labels[0])

        assert len(labels) == 1
        for task, lab in enumerate(labels):
            dropout = DropoutWrapper(task_dropout_p[task], opt['vb_dropout'])
            self.dropout_list.append(dropout)
            out_proj = nn.Linear(self.bert_config.hidden_size, lab)
            self.scoring_list.append(out_proj)

        self.opt = opt
        self._my_init()
        self.set_embed(opt)
        if embedding_matrix is not None and self.use_parse:
            self.treelstm.embedding.weight = nn.Parameter(embedding_matrix)  # set again b/c self._my_init() overwrites it

    def _my_init(self):
        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.bert_config.initializer_range * self.opt['init_ratio'])
            elif isinstance(module, BertLayerNorm):
                # Slightly different from the BERT pytorch version, which should be a bug.
                # Note that it only affects on training from scratch. For detailed discussions, please contact xiaodl@.
                # Layer normalization (https://arxiv.org/abs/1607.06450)
                # support both old/latest version
                if 'beta' in dir(module) and 'gamma' in dir(module):
                    module.beta.data.zero_()
                    module.gamma.data.fill_(1.0)
                else:
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def nbert_layer(self):
        return len(self.bert.encoder.layer)

    def freeze_layers(self, max_n):
        assert max_n < self.nbert_layer()
        for i in range(0, max_n):
            self.freeze_layer(i)

    def freeze_layer(self, n):
        assert n < self.nbert_layer()
        layer = self.bert.encoder.layer[n]
        for p in layer.parameters():
            p.requires_grad = False

    def set_embed(self, opt):
        bert_embeddings = self.bert.embeddings
        emb_opt = opt['embedding_opt']
        if emb_opt == 1:
            for p in bert_embeddings.word_embeddings.parameters():
                p.requires_grad = False
        elif emb_opt == 2:
            for p in bert_embeddings.position_embeddings.parameters():
                p.requires_grad = False
        elif emb_opt == 3:
            for p in bert_embeddings.token_type_embeddings.parameters():
                p.requires_grad = False
        elif emb_opt == 4:
            for p in bert_embeddings.token_type_embeddings.parameters():
                p.requires_grad = False
            for p in bert_embeddings.position_embeddings.parameters():
                p.requires_grad = False

    def forward(self, input_ids, token_type_ids, attention_mask, premise_mask=None, hyp_mask=None, task_id=0,
                bin_parse_as=None, bin_parse_bs=None, parse_as_mask=None, parse_bs_mask=None,
                generic_features=None, domain_features=None):
        all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = all_encoder_layers[-1]
        if self.bert_pooler is not None:
            pooled_output = self.bert_pooler(sequence_output)
        pooled_output = self.dropout_list[task_id](pooled_output)
        logits = self.scoring_list[task_id](pooled_output)

        if self.use_parse:
            parse_embeddings = torch.FloatTensor(len(input_ids), self.stx_parse_dim * 2).to(input_ids.device)
            assert len(bin_parse_as) == len(bin_parse_bs) == len(parse_as_mask) == len(parse_bs_mask)
            for i, (parse_a, parse_b, parse_a_mask, parse_b_mask) in enumerate(zip(bin_parse_as, bin_parse_bs, parse_as_mask, parse_bs_mask)):
                parse_a = parse_a[:parse_a_mask.sum()]
                parse_b = parse_b[:parse_b_mask.sum()]
                t = Tree.from_char_indices(parse_a)
                parse_embeddings[i,:self.stx_parse_dim] = self.treelstm(t)[1]
                t = Tree.from_char_indices(parse_b)
                parse_embeddings[i,self.stx_parse_dim:] = self.treelstm(t)[1]
            logits += self.parse_clf(self.dropout_list[task_id](parse_embeddings))

        if self.use_generic_features:
            # features: bsz * n_features
            generic_feature_embeddings = F.relu(self.generic_feature_proj(generic_features))
            logits += self.generic_feature_clf(self.dropout_list[task_id](generic_feature_embeddings))

        if self.use_domain_features:
            # features: bsz * n_features
            domain_feature_embeddings = F.relu(self.domain_feature_proj(domain_features))
            logits += self.domain_feature_clf(self.dropout_list[task_id](domain_feature_embeddings))

        return logits
