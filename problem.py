#!/usr/bin/env python

"""
    problem.py
"""

from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import sparse
from sklearn import metrics

import torch
from torch.autograd import Variable
from torch.nn import functional as F

from helpers import load_edge_emb

from helpers import read_mpindex_dblp,read_homograph,read_mpindex_yelp,read_mpindex_yago,read_mpindex_cora

# --
# Helper classes

class ProblemLosses:
    @staticmethod
    def multilabel_classification(preds, targets):
        return F.multilabel_soft_margin_loss(preds, targets)
    
    @staticmethod
    def classification(preds, targets):
        return F.cross_entropy(preds, targets)
        #return F.nll_loss(preds, targets)
        #return F.multi_margin_loss(preds, targets,margin=0.2)
  

      

    @staticmethod
    def regression_mae(preds, targets):
        return F.l1_loss(preds, targets)
        
    # @staticmethod
    # def regression_mse(preds, targets):
    #     return F.mse_loss(preds - targets)


class ProblemMetrics:
    @staticmethod
    def multilabel_classification(y_true, y_pred):
        y_pred = (y_pred > 0.5).astype(int)
        return {
            "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
            "micro" : float(metrics.f1_score(y_true, y_pred, average="micro")),
            "macro" : float(metrics.f1_score(y_true, y_pred, average="macro")),
        }
    
    @staticmethod
    def classification(y_true, y_pred):
        y_pred = np.argmax(y_pred, axis=1)
        #print(np.unique(y_true),np.unique(y_pred))
        return {
            "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
            "micro" : float(metrics.f1_score(y_true, y_pred, average="micro")),
            "macro" : float(metrics.f1_score(y_true, y_pred, average="macro")),
        }
        # return (y_pred == y_true.squeeze()).mean()
    
    @staticmethod
    def regression_mae(y_true, y_pred):
        return float(np.abs(y_true - y_pred).mean())


# --
# Problem definition

read_feat_lookup = {
    "dblp":read_mpindex_dblp,
    "yelp":read_mpindex_yelp,
    "yago":read_mpindex_yago,
    "cora":read_mpindex_cora,
}

class NodeProblem(object):
    def __init__(self, problem_path, problem, schemes, device,train_per, K=10, input_edge_dims = 128,):
        
        # print('NodeProblem: loading started')

        features, labels, folds = read_feat_lookup[problem](path=problem_path,train_per=train_per)

        # self.edge_neighs = dict()
        # with np.load("{}edge_neighs_{}_{}.npz".format(problem_path,K, input_edge_dims)) as data:
        #     for s in schemes:
        # #         self.edge_neighs[s] = data[s]
        # self.node_neighs = dict()
        # with np.load("{}node_neighs_{}_{}.npz".format(problem_path,K, input_edge_dims)) as data:
        #     for s in schemes:
        #         self.node_neighs[s] = data[s]
        self.node2edge_idxs = dict()
        with np.load("{}node2edge_idxs_{}_{}.npz".format(problem_path,K, input_edge_dims)) as data:
            for s in schemes:
                self.node2edge_idxs[s] = data[s]
        self.edge_embs = dict()
        with np.load("{}edge_embs_{}_{}.npz".format(problem_path,K, input_edge_dims)) as data:
            for s in schemes:
                self.edge_embs[s] = data[s]
        # print(data[s].shape)
        # self.edge2node_idxs = dict()
        # with np.load("{}edge2node_idxs_{}_{}.npz".format(problem_path,K, input_edge_dims)) as data:
        #     for s in schemes:
        #         self.edge2node_idxs[s] = data[s]

        self.edge_node_adjs = dict()
        with np.load("{}edge_node_adjs_{}_{}.npz".format(problem_path, K, input_edge_dims)) as data:
            for s in schemes:
                self.edge_node_adjs[s] = data[s]

        self.task      = 'classification'
        self.n_classes = int(max(labels)+1) # !!

        #input: features, homograph, edge embedding
        if features.shape[1]>1:
            # self.feats = np.pad(features,((0,1),(0,0)),'constant')
            self.feats = features
            pass
        else:
            self.feats = np.eye(features.shape[0])


        self.schemes=schemes

        self.folds     = folds
        self.targets   = labels

        self.feats_dim = self.feats.shape[1] if self.feats is not None else None
        self.edge_dim = self.edge_embs[schemes[0]].shape[1]
        self.n_nodes   = features.shape[0]

        #self.homo_adj, self.homo_feat = read_homograph(path=problem_path,problem=problem)

        self.device      = device
        self.__to_torch()
        
        self.nodes = {
            "train" : self.folds ['train'],
            "val"   : self.folds ['val'],
            "test"  : self.folds ['test'],
        }
        
        self.loss_fn = getattr(ProblemLosses, self.task)
        self.metric_fn = getattr(ProblemMetrics, self.task)
        
        # print('NodeProblem: loading finished')
    
    def __to_torch(self):
        if self.feats is not None:
            self.feats = torch.FloatTensor(self.feats)

        # for i in self.edge_neighs:
        #      self.edge_neighs[i] = torch.from_numpy(self.edge_neighs[i]).long()
        # for i in self.node_neighs:
        #      self.node_neighs[i] = torch.from_numpy(self.node_neighs[i]).long()
        for i in self.node2edge_idxs:
            self.node2edge_idxs[i] = torch.from_numpy(self.node2edge_idxs[i]).long()
        for i in self.edge_embs:
            self.edge_embs[i] = torch.from_numpy(self.edge_embs[i]).float()
            # print(self.edge_embs[i].shape)
        # for i in self.edge2node_idxs:
        #      self.edge2node_idxs[i] = torch.from_numpy(self.edge2node_idxs[i]).long()

        for i in self.edge_node_adjs:
            self.edge_node_adjs[i] = torch.from_numpy(self.edge_node_adjs[i]).long()
        # if not sparse.issparse(self.adj):
        # if self.device!="cpu":
        #     for i in self.edge_neighs:
        #         self.edge_neighs[i]= self.edge_neighs[i].to(self.device)
        #     for i in self.node_neighs:
        #         self.node_neighs[i]=self.node_neighs[i].to(self.device)
        #     for i in self.node2edge_idxs:
        #         self.node2edge_idxs[i]=self.node2edge_idxs[i].to(self.device)
        #     for i in self.edge_embs:
        #         self.edge_embs[i]=self.edge_embs[i].to(self.device)
        #     for i in self.edge2node_idxs:
        #         self.edge2node_idxs[i]=self.edge2node_idxs[i].to(self.device).detatch()
        #     print('GPU memory allocated: ', torch.cuda.memory_allocated() / 1000 / 1000 / 1000)
        #         # #self.homo_adj = self.homo_adj.to(self.device)
        #         # #self.homo_feat = self.homo_feat.to(self.device)
        #         # for i in self.edge_emb:
        #         #     if torch.is_tensor(self.edge_emb[i]):
        #         #         pass
        #         #         self.edge_emb[i] = self.edge_emb[i].to(self.device)
        #     if self.feats is not None:
        #         self.feats = self.feats.to(self.device)
        #     print('GPU memory allocated: ', torch.cuda.memory_allocated() / 1000 / 1000 / 1000)

    def __batch_to_torch(self, mids, targets):
        """ convert batch to torch """
        mids = Variable(torch.LongTensor(mids))
        
        if self.task == 'multilabel_classification':
            targets = Variable(torch.FloatTensor(targets))
        elif self.task == 'classification':
            targets = Variable(torch.LongTensor(targets))
        elif 'regression' in self.task:
            targets = Variable(torch.FloatTensor(targets))
        else:
            raise Exception('NodeDataLoader: unknown task: %s' % self.task)
        
        if self.device!="cpu":
            mids, targets = mids.to(self.device), targets.to(self.device)
        
        return mids, targets
    
    def iterate(self, mode, batch_size=512, shuffle=False):
        nodes = self.nodes[mode]
        
        idx = np.arange(nodes.shape[0])
        if shuffle:
            idx = np.random.permutation(idx)
        
        n_chunks = idx.shape[0] // batch_size + 1
        for chunk_id, chunk in enumerate(np.array_split(idx, n_chunks)):
            mids = nodes[chunk]
            targets = self.targets[mids].reshape(-1,1)
            mids, targets = self.__batch_to_torch(mids, targets)
            yield mids, targets, chunk_id / n_chunks


class ReadCosSim(object):
    def __init__(self, problem_path, problem, schemes, device,train_per, K=10, input_edge_dims = 128,):
        # print('ReadCosSim: loading started')

        # self.edge_neighs = dict()
        # with np.load("{}edge_neighs_{}_{}.npz".format(problem_path,K, input_edge_dims)) as data:
        #     for s in schemes:
        #         self.edge_neighs[s] = data[s]
        # self.node_neighs = dict()
        # with np.load("{}node_neighs_{}_{}.npz".format(problem_path,K, input_edge_dims)) as data:
        #     for s in schemes:
        #         self.node_neighs[s] = data[s]
        self.node2edge_idxs = dict()
        with np.load("{}node2edge_idxs_{}_{}_cos.npz".format(problem_path,K, input_edge_dims)) as data:
            for s in schemes:
                self.node2edge_idxs[s] = data[s]
        self.edge_embs = dict()
        with np.load("{}edge_embs_{}_{}_cos.npz".format(problem_path,K, input_edge_dims)) as data:
            for s in schemes:
                self.edge_embs[s] = data[s]
        # print(data[s].shape)
        # self.edge2node_idxs = dict()
        # with np.load("{}edge2node_idxs_{}_{}.npz".format(problem_path,K, input_edge_dims)) as data:
        #     for s in schemes:
        #         self.edge2node_idxs[s] = data[s]

        self.edge_node_adjs = dict()
        with np.load("{}edge_node_adjs_{}_{}_cos.npz".format(problem_path, K, input_edge_dims)) as data:
            for s in schemes:
                self.edge_node_adjs[s] = data[s]

        self.device      = device
        self.__to_torch()
        
        # print('ReadCosSim: loading finished')
    
    def __to_torch(self):

        # for i in self.edge_neighs:
        #      self.edge_neighs[i] = torch.from_numpy(self.edge_neighs[i]).long()
        # for i in self.node_neighs:
        #      self.node_neighs[i] = torch.from_numpy(self.node_neighs[i]).long()
        for i in self.node2edge_idxs:
            self.node2edge_idxs[i] = torch.from_numpy(self.node2edge_idxs[i]).long()
        for i in self.edge_embs:
            self.edge_embs[i] = torch.from_numpy(self.edge_embs[i]).float()
            # print(self.edge_embs[i].shape)
        # for i in self.edge2node_idxs:
        #      self.edge2node_idxs[i] = torch.from_numpy(self.edge2node_idxs[i]).long()

        for i in self.edge_node_adjs:
            self.edge_node_adjs[i] = torch.from_numpy(self.edge_node_adjs[i]).long()