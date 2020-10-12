#!/usr/bin/env python

"""
    nn_modules.py
"""
import os
import psutil
import sys
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import gc

import numpy as np
from scipy import sparse
from helpers import to_numpy

import inspect


# from gpu_mem_track import  MemTracker

def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())


def cpuStats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)


# --
# Samplers

class UniformNeighborSampler(object):
    """
        Samples from a "dense 2D edgelist", which looks like
        
            [
                [0, 1, 2, ..., 4],
                [0, 0, 5, ..., 10],
                ...
            ]
        
        stored as torch.LongTensor.

        Adj[a1,a2]=id, where the edge embedding of edge (a1,a2) is stored at emb[id]

        If a node does not have a neighbor, sample itself n_sample times and
        return emb[0]. * emb[0] is the padding zero embedding.

        We didn't apply the optimization from GraphSage
    """

    def __init__(self):
        pass

    def __call__(self, adj, ids, n_samples=16):
        cuda = adj.is_cuda

        neigh = []
        mask = []
        for v in ids:
            nonz = torch.nonzero(adj[v]).view(-1)
            # if (len(nonz) == 0):
            # no neighbor, only sample from itself
            # for edge embedding... PADDING with all-zero embedding at edge_emb[0]
            # if cuda:
            # neigh.append(torch.cuda.LongTensor([v]).repeat(n_samples))
            # mask.append(torch.cuda.LongTensor([1]).repeat(n_samples))
            # else:
            # neigh.append(torch.LongTensor([v]).repeat(n_samples))
            # mask.append(torch.LongTensor([1]).repeat(n_samples))
            # else:
            idx = np.random.choice(nonz.shape[0], n_samples)
            neigh.append(nonz[idx])
        mask = torch.zeros((ids.shape[0], n_samples)).to(ids.device)
        neigh = torch.stack(neigh).long().view(-1)
        edges = adj[
            ids.view(-1, 1).repeat(1, n_samples).view(-1),
            neigh]
        return neigh, edges, mask


class SpUniformNeighborSampler(object):
    """
        Samples from a "sparse 2D edgelist", which looks like

            [
                [0, 1, 2, ..., 4],
                [0, 0, 5, ..., 10],
                ...
            ]

        stored as torch.LongTensor.

        Adj[a1,a2]=id, where the edge embedding of edge (a1,a2) is stored at emb[id]

        If a node does not have a neighbor, sample itself n_sample times and
        return emb[0]. * emb[0] is the padding zero embedding.

        We didn't apply the optimization from GraphSage
    """

    def __init__(self):
        pass

    def __call__(self, adj, ids, n_samples=16):

        cuda = adj.is_cuda

        nonz = adj._indices()
        values = adj._values()

        mask = []
        neigh = []
        edges = []
        for v in ids:
            n = torch.nonzero(nonz[0, :] == v).view(-1)
            # if (len(n) == 0):
            #    # no neighbor, only sample from itself
            #    # for edge embedding... PADDING with all-zero embedding at edge_emb[0]
            #    if cuda:
            #        neigh.append(torch.cuda.LongTensor([v]).repeat(n_samples))
            #        edges.append(torch.cuda.LongTensor([0]).repeat(n_samples))
            #        mask.append(torch.cuda.LongTensor([1]).repeat(n_samples))
            #    else:
            #        neigh.append(torch.LongTensor([v]).repeat(n_samples))
            #        edges.append(torch.LongTensor([0]).repeat(n_samples))
            #        mask.append(torch.LongTensor([1]).repeat(n_samples))
            # else:
            # np.random.choice(nonz.shape[0], n_samples)
            if True:
                # n.shape[0] >= n_samples:
                idx = torch.randint(0, n.shape[0], (n_samples,))

                neigh.append(nonz[1, n[idx]])
                edges.append(values[n[idx]])
                if cuda:
                    mask.append(torch.cuda.LongTensor([0]).repeat(n_samples))
                else:
                    mask.append(torch.LongTensor([0]).repeat(n_samples))

            else:

                if cuda:
                    neigh.append(torch.cat([nonz[1, n], torch.cuda.LongTensor([v]).repeat(n_samples - n.shape[0])]))
                    edges.append(torch.cat([values[n], torch.cuda.LongTensor([0]).repeat(n_samples - n.shape[0])]))
                    mask.append(torch.cat([torch.cuda.LongTensor([0]).repeat(n.shape[0]),
                                           torch.cuda.LongTensor([1]).repeat(n_samples - n.shape[0])]))
                else:
                    neigh.append(torch.cat([nonz[1, n], torch.LongTensor([v]).repeat(n_samples - n.shape[0])]))
                    edges.append(torch.cat([values[n], torch.LongTensor([0]).repeat(n_samples - n.shape[0])]))
                    mask.append(torch.cat([torch.LongTensor([0]).repeat(n.shape[0]),
                                           torch.LongTensor([1]).repeat(n_samples - n.shape[0])]))

        neigh = torch.stack(neigh).long().view(-1)
        edges = torch.stack(edges).long().view(-1)
        mask = torch.stack(mask).float()

        return neigh, edges, mask


class DenseMask(object):
    """
        Samples from a "sparse 2D edgelist", which looks like

            [
                [0, 1, 2, ..., 4],
                [0, 0, 5, ..., 10],
                ...
            ]

        stored as torch.LongTensor.

        Adj[a1,a2]=id, where the edge embedding of edge (a1,a2) is stored at emb[id]

        If a node does not have a neighbor, sample itself n_sample times and
        return emb[0]. * emb[0] is the padding zero embedding.

        We didn't apply the optimization from GraphSage
    """

    def __init__(self):
        pass

    def __call__(self, adj, ids, n_samples=16):

        cuda = adj.is_cuda

        neigh = []
        edges = adj[ids]
        if cuda:
            mask = torch.where(adj == 0, torch.cuda.FloatTensor([0]), torch.cuda.FloatTensor([1]))
        else:
            mask = torch.where(adj == 0, torch.FloatTensor([0]), torch.FloatTensor([1]))

        return neigh, edges, mask[ids]


# --
# Preprocessers

class IdentityPrep(nn.Module):
    def __init__(self, input_dim, n_nodes=None, embedding_dim=64):
        """ Example of preprocessor -- doesn't do anything """
        super(IdentityPrep, self).__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.embedding_dim = input_dim
    def forward(self, ids, feats, layer_idx=0):
        return feats


class NodeEmbeddingPrep(nn.Module):
    def __init__(self, input_dim, n_nodes, pre_trained=None, embedding_dim=64):
        """ adds node embedding """
        super(NodeEmbeddingPrep, self).__init__()

        self.n_nodes = n_nodes
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=n_nodes + 1, embedding_dim=embedding_dim)
        self.fc = nn.Linear(embedding_dim, embedding_dim)  # Affine transform, for changing scale + location

        if pre_trained is not None:
            assert pre_trained.shape[1] == self.embedding_dim
            self.embedding.from_pretrained(pre_trained, padding_idx=None, freeze=False)
            print('loaded from pre-trained embeddings')

    @property
    def output_dim(self):
        if self.input_dim:
            return self.input_dim + self.embedding_dim
        else:
            return self.embedding_dim

    def forward(self, ids, feats, layer_idx=0):
        if layer_idx > 0:
            embs = self.embedding(ids)
        else:
            # Don't look at node's own embedding for prediction, or you'll probably overfit a lot
            embs = self.embedding(Variable(ids.clone().data.zero_() + self.n_nodes))

        embs = self.fc(embs)
        if self.input_dim and feats is not None:
            return torch.cat([feats, embs], dim=1)
        else:
            return embs


class LinearPrep(nn.Module):
    def __init__(self, input_dim, n_nodes, embedding_dim=64):
        """ adds node embedding """
        super(LinearPrep, self).__init__()
        self.fc = nn.Linear(input_dim, embedding_dim, bias=False)
        self.embedding_dim = embedding_dim
        self.output_dim = embedding_dim

    def forward(self, ids, feats, layer_idx=0):
        return self.fc(feats)


# --
# Aggregators

class AggregatorMixin(object):
    @property
    def output_dim(self):
        tmp = torch.zeros((1, self.output_dim_))
        return self.combine_fn([tmp, tmp]).size(1)


class MeanAggregator(nn.Module, AggregatorMixin):
    def __init__(self, input_dim, output_dim, activation, combine_fn=lambda x: torch.cat(x, dim=1)):
        super(MeanAggregator, self).__init__()

        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(input_dim, output_dim, bias=False)

        self.output_dim_ = output_dim
        self.activation = activation
        self.combine_fn = combine_fn

    def forward(self, x, neibs):
        agg_neib = neibs.view(x.size(0), -1, neibs.size(1))  # !! Careful
        agg_neib = agg_neib.mean(dim=1)  # Careful

        out = self.combine_fn([self.fc_x(x), self.fc_neib(agg_neib)])
        if self.activation:
            out = self.activation(out)

        return out


class SumAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, edge_dim, activation, hidden_dim=128,
                 dropout=0.5, alpha=0.8, attn_dropout=0,
                 concat_node=True, concat_edge=True, batchnorm=False):
        super(SumAggregator, self).__init__()

        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(input_dim, output_dim, bias=False)

        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = batchnorm
        self.output_dim = output_dim
        self.activation = activation
        self.concat_node = concat_node
        if concat_node:
            self.output_dim *=2
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(self.output_dim)

    def forward(self, x, neibs, edge_emb, mask):
        # Unweighted average of neighbors
        agg_neib = neibs.view(x.size(0), -1, neibs.size(1))
        agg_neib = torch.sum(agg_neib, dim=1)

        if self.concat_node:
            out = torch.cat([self.fc_x(x), self.fc_neib(agg_neib)], dim=1)
        else:
            out = self.fc_x(x) + self.fc_neib(agg_neib)

        if self.batchnorm:
            out = self.bn(out)

        out = self.dropout(out)

        if self.activation:
            out = self.activation(out)

        return out

class PoolAggregator(nn.Module, AggregatorMixin):
    def __init__(self, input_dim, output_dim, pool_fn, activation, hidden_dim=512,
                 combine_fn=lambda x: torch.cat(x, dim=1)):
        super(PoolAggregator, self).__init__()

        self.mlp = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU()
        ])
        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(hidden_dim, output_dim, bias=False)

        self.output_dim_ = output_dim
        self.activation = activation
        self.pool_fn = pool_fn
        self.combine_fn = combine_fn

    def forward(self, x, neibs):
        h_neibs = self.mlp(neibs)
        agg_neib = h_neibs.view(x.size(0), -1, h_neibs.size(1))
        agg_neib = self.pool_fn(agg_neib)

        out = self.combine_fn([self.fc_x(x), self.fc_neib(agg_neib)])
        if self.activation:
            out = self.activation(out)

        return out


class MaxPoolAggregator(PoolAggregator):
    def __init__(self, input_dim, output_dim, activation, hidden_dim=512, combine_fn=lambda x: torch.cat(x, dim=1)):
        super(MaxPoolAggregator, self).__init__(**{
            "input_dim": input_dim,
            "output_dim": output_dim,
            "pool_fn": lambda x: x.max(dim=1)[0],
            "activation": activation,
            "hidden_dim": hidden_dim,
            "combine_fn": combine_fn,
        })


class MeanPoolAggregator(PoolAggregator):
    def __init__(self, input_dim, output_dim, activation, hidden_dim=512, combine_fn=lambda x: torch.cat(x, dim=1)):
        super(MeanPoolAggregator, self).__init__(**{
            "input_dim": input_dim,
            "output_dim": output_dim,
            "pool_fn": lambda x: x.mean(dim=1),
            "activation": activation,
            "hidden_dim": hidden_dim,
            "combine_fn": combine_fn,
        })


class LSTMAggregator(nn.Module, AggregatorMixin):
    def __init__(self, input_dim, output_dim, activation,
                 hidden_dim=512, bidirectional=False, combine_fn=lambda x: torch.cat(x, dim=1)):
        super(LSTMAggregator, self).__init__()
        assert not hidden_dim % 2, "LSTMAggregator: hiddem_dim % 2 != 0"

        self.lstm = nn.LSTM(input_dim, hidden_dim // (1 + bidirectional), bidirectional=bidirectional, batch_first=True)
        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(hidden_dim, output_dim, bias=False)

        self.output_dim_ = output_dim
        self.activation = activation
        self.combine_fn = combine_fn

    def forward(self, x, neibs):
        x_emb = self.fc_x(x)

        agg_neib = neibs.view(x.size(0), -1, neibs.size(1))
        agg_neib, _ = self.lstm(agg_neib)
        agg_neib = agg_neib[:, -1, :]  # !! Taking final state, but could do something better (eg attention)
        neib_emb = self.fc_neib(agg_neib)

        out = self.combine_fn([x_emb, neib_emb])
        if self.activation:
            out = self.activation(out)

        return out


class AttentionAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, edge_dim, activation, hidden_dim=128,
                 dropout=0.5, alpha=0.8, attn_dropout=0,
                 concat_node=True, concat_edge=True, batchnorm=False):
        super(AttentionAggregator, self).__init__()

        self.att = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        ])
        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(input_dim, output_dim, bias=False)

        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = batchnorm
        self.output_dim = output_dim
        self.activation = activation
        self.concat_node = concat_node
        if concat_node:
            self.output_dim *=2
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(self.output_dim)

    def forward(self, x, neibs, edge_emb, mask):
        # Compute attention weights
        neib_att = self.att(neibs)
        x_att = self.att(x)
        neib_att = neib_att.view(x.size(0), -1, neib_att.size(1))
        x_att = x_att.view(x_att.size(0), x_att.size(1), 1)
        # ws = F.softmax(torch.bmm(neib_att, x_att).squeeze())

        ws = torch.bmm(neib_att, x_att).squeeze()
        if mask is not None:
            ws += -9999999 * mask
        ws = F.softmax(ws, dim=1)

        # Weighted average of neighbors
        agg_neib = neibs.view(x.size(0), -1, neibs.size(1))
        agg_neib = torch.sum(agg_neib * ws.unsqueeze(-1), dim=1)

        if self.concat_node:
            out = torch.cat([self.fc_x(x), self.fc_neib(agg_neib)], dim=1)
        else:
            out = self.fc_x(x) + self.fc_neib(agg_neib)

        if self.batchnorm:
            out = self.bn(out)

        out = self.dropout(out)

        if self.activation:
            out = self.activation(out)

        return out


class EdgeEmbAttentionAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, edge_dim, activation, dropout=0.5, alpha=0.8,
                 concat_node=True, concat_edge=True, batchnorm=False):
        super(EdgeEmbAttentionAggregator, self).__init__()
        self.input_dim = input_dim
        self.edge_dim = edge_dim
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = batchnorm
        self.alpha = alpha
        self.concat_node = concat_node
        if concat_node:
            self.output_dim = 2 * output_dim
        else:
            self.output_dim = output_dim
        if concat_edge:
            self.output_dim += edge_dim
        self.concat_edge = concat_edge

        self.activation = activation
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        W = nn.Parameter(torch.zeros(size=(input_dim, output_dim)))
        nn.init.xavier_uniform_(W.data, gain=1.414)
        self.register_parameter('W', W)

        W2 = nn.Parameter(torch.zeros(size=(input_dim, output_dim)))
        nn.init.xavier_uniform_(W2.data, gain=1.414)
        self.register_parameter('W2', W2)

        a = nn.Parameter(torch.zeros(size=(2 * output_dim + edge_dim, 1)))
        nn.init.xavier_uniform_(a.data, gain=1.414)
        self.register_parameter('a', a)

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, input, neigh_feat, edge_emb, mask):
        # Compute attention weights
        N = input.size()[0]

        x = torch.mm(input, self.W)
        neighs = torch.mm(neigh_feat, self.W2)

        n_sample = int(neighs.shape[0] / x.shape[0])

        a_input = torch.cat([x.repeat(1, n_sample).view(N, n_sample, -1),
                             neighs.view(N, n_sample, -1),
                             edge_emb.view(N, n_sample, -1)], dim=2)

        e = self.leakyrelu(torch.matmul(a_input, self.a))
        # e += -9999999 * mask
        attention = F.softmax(e, dim=1)
        attention = attention.view(N, 1, n_sample)
        # attention = attention.squeeze(2)
        attention = self.dropout(attention)

        # h_prime = [torch.matmul(attention[i], neigh_feat.view(N, n_sample, -1)[i]) for i in range(N)]
        h_prime = torch.bmm(attention, neighs.view(N, n_sample, -1)).squeeze()

        h_prime = self.dropout(h_prime)

        if self.batchnorm:
            h_prime = self.bn(h_prime)

        if self.concat_node:
            output = torch.cat([x, h_prime], dim=1)
        else:
            output = h_prime + x

        if self.concat_edge:
            e = torch.bmm(attention, edge_emb.view(N, n_sample, -1)).squeeze()
            output = torch.cat([output, e], dim=1)
        if self.activation:
            output = self.activation(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' + ' + str(self.edge_dim) \
               + ' -> ' + str(self.output_dim) + ')'


class AttentionAggregator2(nn.Module):
    def __init__(self, input_dim, output_dim, edge_dim, activation, hidden_dim=64,
                 dropout=0.5, attn_dropout=0,
                 concat_node=True, concat_edge=True, batchnorm=False):
        super(AttentionAggregator2, self).__init__()
        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.hidden_dim = hidden_dim
        self.att = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        ])

        self.att2 = nn.Sequential(*[
            nn.Linear(input_dim + edge_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        ])

        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(input_dim + edge_dim, output_dim, bias=False)
        self.concat_node = concat_node

        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = batchnorm

        if concat_node:
            self.output_dim = output_dim * 2
        else:
            self.output_dim = output_dim
        self.activation = activation

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(self.output_dim)

        # self.apply(weight_init)

    def forward(self, x, neibs, edge_emb, mask):
        # Compute attention weights
        neibs = torch.cat([neibs, edge_emb], dim=1)

        neib_att = self.att2(neibs)
        x_att = self.att(x)

        neib_att = neib_att.view(x.size(0), -1, neib_att.size(1))
        x_att = x_att.view(x_att.size(0), x_att.size(1), 1)

        ws = torch.bmm(neib_att, x_att).squeeze()

        import math
        ws /= math.sqrt(self.hidden_dim)
        if mask is not None:
            ws += -9999999 * mask
        # ws = F.leaky_relu(ws)
        ws = F.softmax(ws, dim=1)

        # dropout for attention coefficient
        ws = self.attn_dropout(ws)
        # ws = F.normalize(ws,p=1,dim=1)

        # Weighted average of neighbors
        agg_neib = neibs.view(x.size(0), -1, neibs.size(1))
        agg_neib = torch.bmm(ws.view(x.size(0), 1, -1), agg_neib).squeeze()

        if self.concat_node:
            out = torch.cat([self.fc_x(x), self.fc_neib(agg_neib)], dim=1)
        else:
            out = self.fc_x(x) + self.fc_neib(agg_neib)

        if self.batchnorm:
            out = self.bn(out)

        out = self.dropout(out)

        if self.activation:
            out = self.activation(out)

        return out


class DenseAttentionAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, edge_dim, activation,
                 adj, edge_emb, hidden_dim=32,
                 dropout=0.5,
                 concat_node=True, concat_edge=True, batchnorm=False):
        super(DenseAttentionAggregator, self).__init__()

        self.adj = adj
        self.edge_emb = edge_emb

        self.att_x = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            # nn.Tanh(),
        ])

        self.att_neigh = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            # nn.Tanh(),
        ])

        self.fc_value = nn.Linear(input_dim, output_dim)

        self.fc_x = nn.Linear(input_dim, output_dim)
        # self.fc_neib = nn.Linear(input_dim, output_dim)
        self.concat_node = concat_node

        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = batchnorm

        if concat_node:
            self.output_dim = output_dim * 2
        else:
            self.output_dim = output_dim
        self.activation = activation

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(self.output_dim)

    def forward(self, x, neigh, batch=512):
        '''
        :param x: (n_nodes,input_dim)
        :param neibs: (n_nodes,input_dim)
        :param edge_emb: (N*n_nodes,edge_dim)
        :param mask: (N, n_nodes)
        :return:
        '''
        neib_att = self.att_neigh(neigh)
        value = self.fc_value(neigh)
        result = []
        adjs = torch.split(self.adj, batch, dim=0)
        for chunk_id, chunk in enumerate(torch.split(x, batch, dim=0)):
            N = chunk.shape[0]
            edges = self.edge_emb[adjs[chunk_id].view(-1)]

            x_att = self.att_x(chunk)
            # edge_att = self.att_edge(edge_emb)

            ws = x_att.mm(neib_att.t())  # +edge_att.view(N,-1)
            # ws = x_att+neib_att.t()
            ws = F.leaky_relu(ws)
            zero_vec = -9e15 * torch.ones_like(ws)
            ws = torch.where(adjs[chunk_id] > 0, ws, zero_vec)
            ws = F.softmax(ws, dim=1)
            ws = F.dropout(ws, 0.3, training=self.training)

            # Weighted average of neighbors
            agg_neib = torch.mm(ws, value)
            # agg_neib = F.sigmoid(agg_neib)
            # agg_edge = edge_emb.view(N, -1, edge_emb.size(-1))
            # agg_edge = torch.sum(agg_edge * ws.unsqueeze(-1), dim=1)

            if self.concat_node:
                out = torch.cat([self.fc_x(chunk), agg_neib], dim=1)
            else:
                out = self.fc_x(chunk) + agg_neib
                out = F.elu(out)
            if self.batchnorm:
                out = self.bn(out)
            result.append(out)
        result = torch.cat(result, dim=0)
        result = self.dropout(result)
        return result


class DenseEdgeAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, edge_dim, activation,
                 adj, edge_emb, hidden_dim=32,
                 dropout=0.5,
                 concat_node=True, concat_edge=True, batchnorm=False):
        super(DenseEdgeAggregator, self).__init__()

        self.adj = adj
        self.edge_emb = edge_emb

        self.att_x = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            # nn.Tanh(),
        ])

        self.att_neigh = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            # nn.Tanh(),
        ])

        self.att_edge = nn.Sequential(*[
            nn.Linear(edge_dim, hidden_dim, bias=True),
            nn.Tanh(),
            # nn.Linear(hidden_dim, hidden_dim, bias=True),
            # nn.Tanh(),
        ])

        self.fc_value = nn.Linear(input_dim, output_dim)
        self.fc_edge = nn.Linear(edge_dim, output_dim)

        self.fc_x = nn.Linear(input_dim, output_dim)
        # self.fc_neib = nn.Linear(input_dim, output_dim)
        self.concat_node = concat_node

        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = batchnorm

        if concat_node:
            self.output_dim = output_dim * 2
        else:
            self.output_dim = output_dim
        self.activation = activation

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(self.output_dim)

    def forward(self, x, neigh, batch=4):
        '''
        :param x: (n_nodes,input_dim)
        :param neibs: (n_nodes,input_dim)
        :param edge_emb: (N*n_nodes,edge_dim)
        :param mask: (N, n_nodes)
        :return:
        '''
        # from gpu_mem_track import MemTracker

        # frame = inspect.currentframe()
        # gpu_tracker = MemTracker(frame)
        # cpuStats()
        # memReport()
        neib_att = self.att_neigh(neigh)
        value = self.fc_value(neigh)
        result = []
        ids = torch.arange(x.shape[0])
        # gpu_tracker.track()
        for i, chunk_ids in enumerate(torch.split(ids, batch, dim=0)):
            # gpu_tracker.track()
            chunk = x[chunk_ids]
            edges = self.edge_emb[self.adj[chunk_ids].view(-1)].to(x.device)
            # gpu_tracker.track()
            N = chunk.shape[0]
            k = edges.shape[0] // N
            edges = edges.reshape(N, k, -1)
            # print(edges.shape)
            # gpu_tracker.track()
            x_att = self.att_x(chunk)
            # edge_att = self.att_edge(edge_emb)
            # gpu_tracker.track()
            ws = x_att.mm(neib_att.t()) + torch.bmm(self.att_edge(edges), x_att.view(N, -1, 1)).squeeze()
            # ws = x_att+neib_att.t()
            ws = F.leaky_relu(ws)
            zero_vec = -9e15 * torch.ones_like(ws, requires_grad=False)
            ws = torch.where(self.adj[chunk_ids] > 0, ws, zero_vec)
            ws = F.softmax(ws, dim=1)
            # attention = F.dropout(attention, self.dropout, training=self.training)

            # Weighted average of neighbors
            agg_neib = torch.mm(ws, value) + torch.bmm(ws.view(N, 1, k), self.fc_edge(edges)).squeeze()
            # agg_neib = F.sigmoid(agg_neib)
            # agg_edge = edge_emb.view(N, -1, edge_emb.size(-1))
            # agg_edge = torch.sum(agg_edge * ws.unsqueeze(-1), dim=1)
            # gpu_tracker.track()
            if self.concat_node:
                out = torch.cat([self.fc_x(chunk), agg_neib], dim=1)
            else:
                out = self.fc_x(chunk) + agg_neib
                out = F.elu(out)
            if self.batchnorm:
                out = self.bn(out)
            result.append(out)
            # del edges
            # del x_att
            # del ws
            # del zero_vec
            # del agg_neib
            # del chunk
            # del out
            # gc.collect()
            # torch.cuda.empty_cache()
            # gpu_tracker.track()
        result = torch.cat(result, dim=0)
        result = self.dropout(result)
        del value
        # gc.collect()
        # torch.cuda.empty_cache()
        # gpu_tracker.track()
        return result


class MRAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, edge_dim, activation, hidden_dim=256,
                 dropout=0.5,
                 concat_node=True, concat_edge=True, batchnorm=False):
        super(MRAggregator, self).__init__()

        self.mlp = nn.Sequential(*[
            # nn.Linear(input_dim, hidden_dim, bias=True),
            # nn.tanh(),
            nn.Linear(hidden_dim, output_dim, bias=True),
            # nn.ReLU()
        ])
        self.fc_x = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_neib = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_edge = nn.Linear(edge_dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim
        self.activation = activation

    def forward(self, x, neibs, edge_emb, mask):
        n_sample = int(neibs.shape[0] / x.shape[0])
        N = x.shape[0]

        # h_neibs = self.mlp(neibs)
        agg_neib = neibs.view(x.size(0), -1, neibs.size(1)) - x.repeat(1, n_sample).view(N, n_sample, -1)
        a = self.fc_x(x.repeat(1, n_sample).view(N, n_sample, -1)) \
            + self.fc_neib(agg_neib) + self.fc_edge(edge_emb.view(N, n_sample, -1))
        a = F.relu(a)

        out = torch.max(a, dim=1)[0].squeeze()

        out = self.mlp(out)

        out = self.dropout(out)
        if self.activation:
            out = self.activation(out)

        return out


class EdgeAggregator(nn.Module):
    def __init__(self, input_dim, edge_dim, activation, dropout=0.5, batchnorm=False):
        super(EdgeAggregator, self).__init__()

        self.input_dim = input_dim
        self.edge_dim = edge_dim
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = batchnorm

        W1 = nn.Parameter(torch.zeros(size=(input_dim, edge_dim)))
        nn.init.xavier_uniform_(W1.data, gain=1.414)
        self.register_parameter('W1', W1)

        W2 = nn.Parameter(torch.zeros(size=(edge_dim, edge_dim)))
        nn.init.xavier_uniform_(W2.data, gain=1.414)
        self.register_parameter('W2', W2)

        B = nn.Parameter(torch.zeros(size=(1, edge_dim)))
        nn.init.xavier_uniform_(B.data, gain=1.414)
        self.register_parameter('B', B)

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(self.edge_dim)

    def forward(self, x, neibs, edge_emb, mask):
        # update edge embedding:
        # e = sigma(w1*x+W2*neibs+b) @ e

        # self.W1.to(x.device)
        # self.W2.to(x.device)
        # self.B.to(x.device)

        n = edge_emb.shape[0]
        n_sample = int(edge_emb.shape[0] / x.shape[0])

        x_input = torch.mm(x.repeat(n_sample, 1), self.W1)

        n_input = torch.mm(neibs, self.W1)

        e_input = torch.mm(edge_emb, self.W2)

        a_input = e_input + n_input + x_input + self.B.repeat(n, 1)

        a_input = self.dropout(a_input)

        if self.batchnorm:
            a_input = self.bn(a_input)

        if self.activation:
            a_input = self.activation(a_input)

        emb = a_input * edge_emb

        return emb

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' + ' + str(self.edge_dim) \
               + ' -> ' + str(self.edge_dim) + ')'

class IdEdgeAggregator(nn.Module):
    def __init__(self, input_dim, edge_dim, activation, dropout=0.5, batchnorm=False):
        super(IdEdgeAggregator, self).__init__()

        self.input_dim = input_dim
        self.activation = activation
        self.edge_dim = edge_dim
        self.batchnorm = batchnorm
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, neibs, edge_emb, mask):
        # identical mapping
        # e = sigma(w1*x+W2*neibs+b) @ e
        return edge_emb


class ResEdge(nn.Module):
    def __init__(self, input_dim, edge_dim, activation, dropout=0.5, batchnorm=False):
        super(ResEdge, self).__init__()

        self.input_dim = input_dim
        self.edge_dim = edge_dim
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = batchnorm

        W1 = nn.Parameter(torch.zeros(size=(input_dim, edge_dim)))
        nn.init.xavier_uniform_(W1.data, gain=1.414)
        self.register_parameter('W1', W1)

        W2 = nn.Parameter(torch.zeros(size=(edge_dim, edge_dim)))
        nn.init.xavier_uniform_(W2.data, gain=1.414)
        self.register_parameter('W2', W2)

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(self.edge_dim)

    def forward(self, x, neibs, edge_emb, mask):
        # update edge embedding:
        # e = sigma(W1*x+W1*neibs+W2*e) + e

        # n = edge_emb.shape[0]
        # self.W1.to(x.device)
        # self.W2.to(x.device)

        n_sample = int(edge_emb.shape[0] / x.shape[0])

        x_input = torch.mm(x, self.W1).repeat(n_sample, 1)

        n_input = torch.mm(neibs, self.W1)

        e_input = torch.mm(edge_emb, self.W2)

        a_input = e_input + n_input + x_input

        a_input = self.dropout(a_input)

        if self.batchnorm:
            a_input = self.bn(a_input)

        if self.activation:
            a_input = self.activation(a_input)

        emb = a_input + edge_emb

        return emb

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' + ' + str(self.edge_dim) \
               + ' -> ' + str(self.edge_dim) + ')'


class GRUEdge(nn.Module):
    def __init__(self, input_dim, edge_dim, activation, dropout=0.5, batchnorm=False):
        super(GRUEdge, self).__init__()

        self.input_dim = input_dim
        self.edge_dim = edge_dim
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = batchnorm

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(self.edge_dim)

        self.gate1 = nn.Linear(2 * input_dim, edge_dim)
        self.gate2 = nn.Linear(edge_dim, edge_dim)
        self.gate3 = nn.Linear(2 * input_dim, edge_dim)
        self.gate4 = nn.Linear(edge_dim, edge_dim)
        self.fc_x = nn.Linear(2 * input_dim, edge_dim)
        self.fc_edge = nn.Linear(edge_dim, edge_dim)

    def forward(self, x, neibs, edge_emb, mask):
        # update edge embedding:
        # e = sigma(W1*x+W1*neibs+W2*e) + e

        n_sample = int(edge_emb.shape[0] / x.shape[0])
        x_input = torch.cat([x.repeat(n_sample, 1), neibs], dim=1)

        x_prime = self.gate1(x_input)

        e_input = self.gate2(edge_emb)

        update_gate = torch.sigmoid(x_prime + e_input)

        x_prime2 = self.gate3(x_input)

        e_input2 = self.gate4(edge_emb)

        reset_gate = torch.sigmoid(x_prime2 + e_input2)

        h_prime = torch.tanh(self.fc_x(x_input) + reset_gate * self.fc_edge(edge_emb))

        emb = update_gate * edge_emb + (1 - update_gate) * h_prime

        return emb

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' + ' + str(self.edge_dim) \
               + ' -> ' + str(self.edge_dim) + ')'

class MetapathSumLayer(nn.Module):
    """
    metapath sum layer.
    """

    def __init__(self, in_features, n_head=3, alpha=0.8, dropout=0.5, hidden_dim=64, batchnorm=False):
        super(MetapathSumLayer, self).__init__()
        # self.dropout = dropout
        self.input_dim = in_features
        self.output_dim = in_features


    def forward(self, input):
        """
        :param input: tensor(nmeta,N,in_features)
        :return:
        """
        input = input.sum(dim=0).squeeze()

        weight = None

        return input, weight

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'

class MetapathMeanLayer(nn.Module):
    """
    metapath mean layer.
    """

    def __init__(self, in_features, n_head=3, alpha=0.8, dropout=0.5, hidden_dim=64, batchnorm=False):
        super(MetapathMeanLayer, self).__init__()
        # self.dropout = dropout
        self.input_dim = in_features
        self.output_dim = in_features


    def forward(self, input):
        """
        :param input: tensor(nmeta,N,in_features)
        :return:
        """
        input = input.mean(dim=0).squeeze()

        weight = None

        return input, weight

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'

class MetapathPoolingLayer(nn.Module):
    """
    metapath sum layer.
    """

    def __init__(self, in_features, n_head=3, alpha=0.8, dropout=0.5, hidden_dim=64, batchnorm=False):
        super(MetapathPoolingLayer, self).__init__()
        # self.dropout = dropout
        self.input_dim = in_features
        self.output_dim = in_features


    def forward(self, input):
        """
        :param input: tensor(nmeta,N,in_features)
        :return:
        """
        input,_ = input.max(0)

        weight = None

        return input, weight

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'


class MetapathConcatLayer(nn.Module):
    """
    metapath concat layer.
    """

    def __init__(self, in_features, n_head=3, alpha=0.8, dropout=0.5, hidden_dim=64, batchnorm=False):
        super(MetapathConcatLayer, self).__init__()
        # self.dropout = dropout
        self.input_dim = in_features
        self.output_dim = in_features*n_head


    def forward(self, input):
        """
        :param input: tensor(nmeta,N,in_features)
        :return:
        """
        n_meta = input.shape[0]
        input = input.transpose(0, 1)  # tensor(N,nmeta,in_features)
        N = input.size()[0]
        input_dim = input.shape[2]

        input = input.reshape(N,-1)

        weight = None

        return input, weight

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'


class MetapathAttentionLayer(nn.Module):
    """
    metapath attention layer.
    """

    def __init__(self, in_features, n_head=4, alpha=0.8, dropout=0.5, hidden_dim=64, batchnorm=False):
        super(MetapathAttentionLayer, self).__init__()
        # self.dropout = dropout
        self.input_dim = in_features
        self.output_dim = in_features
        self.alpha = alpha
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = batchnorm

        self.att = nn.Sequential(*[
            nn.Linear(in_features, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        ])

        self.mlp = nn.Sequential(*[
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_features),
        ])

        a = nn.Parameter(torch.zeros(size=(hidden_dim, 1)))
        nn.init.xavier_uniform_(a.data, gain=1.414)
        self.register_parameter('a', a)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(self.output_dim)

    def forward(self, input):
        """
        :param input: tensor(nmeta,N,in_features)
        :return:
        """
        n_meta = input.shape[0]
        input = input.transpose(0, 1)  # tensor(N,nmeta,in_features)
        N = input.size()[0]
        input_dim = input.shape[2]

        input = input.contiguous()
        a_input = self.att(input.view(-1, input_dim)) \
            .view(N, n_meta, -1)

        # a_input = torch.cat([input.repeat(1,1,n_meta).view(N, n_meta*n_meta, -1),
        #                      input.repeat(1,n_meta, 1)], dim=2).view(N, -1, 2 * input_dim)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # e: tensor(N,nmeta)
        e = F.softmax(e, dim=1).view(N, 1, n_meta)

        output = torch.bmm(e, input).squeeze()

        output = self.dropout(output)
        output = self.mlp(output)

        if self.batchnorm:
            output = self.bn(output)

        weight = torch.sum(e.view(N, n_meta), dim=0) / N

        return F.relu(output), weight

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'


class MetapathAttentionLayer2(nn.Module):
    """
    metapath attention layer， from HDGI and HAN.
    """

    def __init__(self, in_features, n_head=4, alpha=0.8, dropout=0.5, hidden_dim=64, batchnorm=False):
        super(MetapathAttentionLayer2, self).__init__()
        # self.dropout = dropout
        self.in_features = in_features
        self.out_features = in_features
        self.output_dim = in_features
        out_features = in_features

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.b = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        self.q = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.q.data, gain=1.414)
        self.Tanh = nn.Tanh()

    def forward(self, input):
        """
        :param input: tensor(nmeta,N,in_features)
        :return:
        """
        P = n_meta = input.shape[0]
        N = input.size()[1]
        input_dim = input.shape[2]

        # input = input.transpose(0, 1)  # tensor(N,nmeta,in_features)
        input = input.contiguous().view(-1,input_dim)
        
        h = torch.mm(input, self.W)
        #h=(PN)*F'
        h_prime = self.Tanh(h + self.b.repeat(h.size()[0],1))
        #h_prime=(PN)*F'
        semantic_attentions = torch.mm(h_prime, torch.t(self.q)).view(P,-1)       
        #semantic_attentions = P*N
        N = semantic_attentions.size()[1]
        semantic_attentions = semantic_attentions.mean(dim=1,keepdim=True)
        #semantic_attentions = P*1
        semantic_attentions = F.softmax(semantic_attentions, dim=0)
        # print(semantic_attentions)
        semantic_attentions = semantic_attentions.view(P,1,1)
        semantic_attentions = semantic_attentions.repeat(1,N,self.in_features)
#        print(semantic_attentions)
        #input_embedding = P*N*F
        input_embedding = input.view(P,N,self.in_features)
        
        #h_embedding = N*F
        h_embedding = torch.mul(input_embedding, semantic_attentions)
        h_embedding = torch.sum(h_embedding, dim=0).squeeze()
        
        return h_embedding, semantic_attentions

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class MetapathGateLayer(nn.Module):
    """
    metapath gated attention layer.
    """

    def __init__(self, in_features, n_head=4, alpha=0.8, dropout=0.5, hidden_dim=64, batchnorm=False):
        super(MetapathGateLayer, self).__init__()
        # self.dropout = dropout
        self.input_dim = in_features
        self.output_dim = in_features
        self.alpha = alpha
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = batchnorm

        self.att = nn.Sequential(*[
            nn.Linear(in_features, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, in_features, bias=True),
        ])

        self.mlp = nn.Sequential(*[
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_features),
        ])

        # a = nn.Parameter(torch.zeros(size=(hidden_dim, 1)))
        # nn.init.xavier_uniform_(a.data, gain=1.414)
        # self.register_parameter('a', a)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.gate = nn.Sequential(*[
            nn.Linear(in_features, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, in_features, bias=True),
        ])
        self.update = nn.Sequential(*[
            nn.Linear(in_features, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, in_features, bias=True),
        ])

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(self.output_dim)

    def forward(self, input):
        """
        :param input: tensor(nmeta,N,in_features)
        :return:
        """
        n_meta = input.shape[0]
        # input = input.transpose(0, 1)  # tensor(N,nmeta,in_features)
        N = input.size()[1]
        input_dim = input.shape[2]

        # input = input.contiguous()
        gate_input = F.sigmoid(self.gate(input))
        update_input = F.tanh(self.update(input))
        output = gate_input * update_input

        output = torch.sum(output, dim=0).squeeze()

        # a_input = self.att(input)
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))   #e: tensor(N,nmeta)
        # e = F.softmax(e, dim=1).view(N, 1, n_meta)

        # output = torch.bmm(e, input).squeeze()

        # output = self.dropout(output)
        output = self.mlp(output)

        if self.batchnorm:
            output = self.bn(output)

        # weight = torch.sum(e.view(N, n_meta), dim=0) / N
        weight = None

        return F.relu(output), weight

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'


class MetapathAttentionLayerWBackground(nn.Module):
    """
    metapath attention layer.
    """

    def __init__(self, in_features, n_head=4, alpha=0.8, dropout=0.5, hidden_dim=64, batchnorm=False):
        super(MetapathAttentionLayerWBackground, self).__init__()
        # self.dropout = dropout
        self.input_dim = in_features
        self.output_dim = in_features
        self.alpha = alpha
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = batchnorm

        self.att = nn.Sequential(*[
            nn.Linear(in_features, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        ])

        self.fc_x = nn.Sequential(*[
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),
        ])

        self.fc_back = nn.Sequential(*[
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),
        ])

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(self.out_features)

    def forward(self, input):
        """
        :param input: tensor(nmeta,N,in_features)
        :return:
        """
        n_meta = input.shape[0]
        input = input.transpose(0, 1)  # tensor(N,nmeta,in_features)
        N = input.size()[0]
        input_dim = input.shape[2]

        back = input[:, -1, :]
        input = input[:, :-1, :].contiguous()
        x_att = self.att(input)  # tensor(N,nmeta-1,hidden)
        back_att = self.att(back).view(N, -1, 1)  # tensor(N,hidden,1)

        e = self.leakyrelu(torch.bmm(x_att, back_att).squeeze(2))  # e: tensor(N,nmeta-1)
        e = F.softmax(e, dim=1).view(N, 1, n_meta - 1)

        output = torch.bmm(e, input).squeeze()
        output = self.fc_x(output) + self.fc_back(back)

        output = self.dropout(output)

        if self.batchnorm:
            output = self.bn(output)

        weight = torch.sum(e.view(N, n_meta - 1), dim=0) / N

        return F.relu(output), weight

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'


class MetapathLSTMLayer(nn.Module):
    """
    metapath LSTM layer.
    """

    def __init__(self, in_features, n_head=4, alpha=0.8, dropout=0.5, hidden_dim=512, batchnorm=False,
                 bidirectional=False):
        super(MetapathLSTMLayer, self).__init__()
        # self.dropout = dropout
        self.input_dim = in_features
        self.output_dim = in_features
        self.alpha = alpha
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = batchnorm

        # self.att = nn.sequential(*[
        #    nn.linear(in_features, hidden_dim, bias=false),
        #    nn.tanh(),
        #    nn.linear(hidden_dim, hidden_dim, bias=false),
        # ])
        self.lstm = nn.LSTM(self.input_dim, hidden_dim // (1 + bidirectional), bidirectional=bidirectional,
                            batch_first=True)

        self.mlp = nn.Sequential(*[
            nn.Linear(hidden_dim, in_features),
            nn.ReLU(),
            nn.Linear(in_features, in_features),
        ])

        a = nn.Parameter(torch.zeros(size=(hidden_dim, 1)))
        nn.init.xavier_uniform_(a.data, gain=1.414)
        self.register_parameter('a', a)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(self.out_features)

    def forward(self, input):
        """
        :param input: tensor(nmeta,N,in_features)
        :return:
        """
        n_meta = input.shape[0]
        input = input.transpose(0, 1)  # tensor(N,nmeta,in_features)
        N = input.size()[0]
        input_dim = input.shape[2]

        # input=input.view(x.size(0), -1, neibs.size(1))
        agg_neib, _ = self.lstm(input)
        agg_neib = agg_neib[:, -1, :]  # !! Taking final state, but could do something better (eg attention)

        output = self.dropout(agg_neib)

        if self.batchnorm:
            output = self.bn(output)

        weight = None

        return F.relu(self.mlp(output)), weight

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'


class MetapathGRULayer(nn.Module):
    """
    metapath gated recurrent unit layer.
    """

    def __init__(self, in_features, n_head=4, alpha=0.8, dropout=0.5, hidden_dim=512, batchnorm=False):
        super(MetapathGRULayer, self).__init__()
        # self.dropout = dropout
        self.input_dim = in_features
        self.output_dim = in_features
        self.alpha = alpha
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = batchnorm

        self.mlp = nn.Sequential(*[
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_features),
        ])

        m = torch.zeros(size=(n_head, hidden_dim))  # memory of GRU
        self.register_buffer('m', m)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.gate = nn.Sequential(*[
            nn.Linear(in_features, hidden_dim, bias=True),
        ])
        self.gate2 = nn.Sequential(*[
            nn.Linear(in_features, hidden_dim, bias=True),
        ])
        self.update = nn.Sequential(*[
            nn.Linear(in_features, hidden_dim, bias=True),
        ])
        self.out = nn.Sequential(*[
            nn.Linear(in_features, hidden_dim, bias=True),
        ])

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(self.output_dim)

    def forward(self, input):
        """
        :param input: tensor(nmeta,N,in_features)
        :return:
        """
        n_meta = input.shape[0]
        input = input.transpose(0, 1)  # tensor(N,nmeta,in_features)
        N = input.size()[0]
        input_dim = input.shape[2]
        input = input.contiguous()

        memory = self.m.unsqueeze(0).repeat((N, 1, 1))  # tensor(N,nmeta,hidden)
        memory = torch.sigmoid(self.gate(input)) * memory

        gate_input = torch.sigmoid(self.gate2(input))
        update_input = torch.tanh(self.update(input))

        memory += gate_input * update_input

        output = torch.tanh(memory) * torch.sigmoid(self.out(input))

        output = torch.sum(output, dim=1).squeeze()
        self.m = torch.sum(memory, dim=1).squeeze() / N

        output = self.dropout(output)
        output = self.mlp(output)

        if self.batchnorm:
            output = self.bn(output)

        # weight = torch.sum(e.view(N, n_meta), dim=0) / N
        weight = None

        return F.relu(output), weight

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'


sampler_lookup = {
    "uniform_neighbor_sampler": UniformNeighborSampler,
    "sparse_uniform_neighbor_sampler": SpUniformNeighborSampler,
    "dense_mask": DenseMask,
}

prep_lookup = {
    "identity": IdentityPrep,
    "node_embedding": NodeEmbeddingPrep,
    "linear": LinearPrep,
}

aggregator_lookup = {
    "sum": SumAggregator,
    "mean": MeanAggregator,
    "max_pool": MaxPoolAggregator,
    "mean_pool": MeanPoolAggregator,
    "lstm": LSTMAggregator,
    "attention": AttentionAggregator,
    "attention2": AttentionAggregator2,
    "dense_attention": DenseAttentionAggregator,
    "dense_edge": DenseEdgeAggregator,
    "edge_emb_attn": EdgeEmbAttentionAggregator,
    "MR": MRAggregator,
}

metapath_aggregator_lookup = {
    "attention": MetapathAttentionLayer,
    "attention2": MetapathAttentionLayer2,
    "concat": MetapathConcatLayer,
    "LSTM": MetapathLSTMLayer,
    "gate": MetapathGateLayer,
    "GRU": MetapathGRULayer,
    'sum': MetapathSumLayer,
    'pooling': MetapathPoolingLayer,
    'mean':MetapathMeanLayer,
}

edge_aggregator_lookup = {
    "identity": IdEdgeAggregator,
    "attention": EdgeAggregator,
    # "sum": EdgeSumAggregator,
    "residual": ResEdge,
    "GRU": GRUEdge,
}

import math


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj = adj
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.mm(input, self.weight)
        output = torch.spmm(self.adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
