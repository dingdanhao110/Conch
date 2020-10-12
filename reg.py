from __future__ import division
from __future__ import print_function

from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from lr import LRSchedule

from nn_modules import GraphConvolution, IdEdgeAggregator

class Conch_reg(nn.Module):
    """
    No more samplers.
    Node feature: [N,d]
    Node neighbors: [N,S]
    Edge embedding: [E,e]
    Node edge mapping: [N,S]->E
    Edge neighbors: [E,2S]
    """

    def __init__(self,
                 n_mp,
                 problem,
                 cos_problem,
                 prep_len,
                 n_head,
                 node_layer_specs,
                 edge_layer_specs,
                 aggregator_class,
                 mpaggr_class,
                 edge_aggr_class,
                 prep_class,
                 sampler_class,
                 dropout,
                 batchnorm,
                #  coff=1,
                 attn_dropout=0,
                 bias=False,
                 ):

        super(Conch_reg, self).__init__()
        # self.coff = coff
        # --
        # Input Data
        self.edge_dim = problem.edge_dim
        self.input_dim = problem.feats_dim
        self.n_nodes = problem.n_nodes
        self.n_classes = problem.n_classes
        self.n_head = n_head
        self.bias = bias

        # self.feats
        self.register_buffer('feats', problem.feats)

        # self.edge_neigh_mp
        # for i, key in enumerate(problem.edge_neighs):
        #     self.register_buffer('edge_neigh_{}'.format(i),
        #                          problem.edge_neighs[key])

        # # self.node_neigh_mp
        # for i, key in enumerate(problem.node_neighs):
        #     self.register_buffer('node_neigh_{}'.format(i),
        #                          problem.node_neighs[key])

        # self.node2edge_idx_mp
        for i, key in enumerate(problem.node2edge_idxs):
            self.register_buffer('node2edge_idx_{}'.format(i),
                                 problem.node2edge_idxs[key])

        # self.edge_emb_mp
        for i, key in enumerate(problem.edge_embs):
            self.register_buffer('edge_emb_{}'.format(i),
                                 problem.edge_embs[key])
            # print(problem.edge_embs[key].shape)

        # # self.edge2node_idx_mp
        # for i, key in enumerate(problem.edge2node_idxs):
        #     self.register_buffer('edge2node_idx_{}'.format(i),
        #                          problem.edge2node_idxs[key])

        # self.edge_node_adj_mp
        for i, key in enumerate(problem.edge_node_adjs):
            self.register_buffer('edge_node_adj_{}'.format(i),
                                 problem.edge_node_adjs[key])

        #------cos-------
        # self.cos_node2edge_idx_mp
        for i, key in enumerate(cos_problem.node2edge_idxs):
            self.register_buffer('cos_node2edge_idx_{}'.format(i),
                                 cos_problem.node2edge_idxs[key])

        # self.cos_edge_emb_mp
        for i, key in enumerate(cos_problem.edge_embs):
            self.register_buffer('cos_edge_emb_{}'.format(i),
                                 cos_problem.edge_embs[key])
            # print(cos_problem.edge_embs[key].shape)

        # self.cos_edge_node_adj_mp
        for i, key in enumerate(cos_problem.edge_node_adjs):
            self.register_buffer('cos_edge_node_adj_{}'.format(i),
                                 cos_problem.edge_node_adjs[key])

        # Define network
        self.n_mp = n_mp
        self.depth = len(node_layer_specs)
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.batchnorm = batchnorm

        # No more samplers

        self.train_batch = 999999

        # Prep
        # import numpy as np
        # with open("{}{}.emb".format('data/freebase/', 'MADW_16')) as f:
        #     n_nodes, n_feature = map(int, f.readline().strip().split())
        # print("number of nodes:{}, embedding size:{}".format(n_nodes, n_feature))
        #
        # embedding = np.loadtxt("{}{}.emb".format('data/freebase/', 'MADW_16'),
        #                        dtype=np.float32, skiprows=1)
        # emd_index = {}
        # for i in range(n_nodes):
        #     emd_index[embedding[i, 0]] = i

        # print(emd_index[0])

        # features = np.asarray([embedding[emd_index[i], 1:] for i in range(n_nodes)])
        #
        # assert features.shape[1] == n_feature
        # assert features.shape[0] == n_nodes

        # print(features[0,:])

        # features = torch.FloatTensor(features[:problem.n_nodes,:])
        self.prep = prep_class(input_dim=problem.feats_dim, n_nodes=problem.n_nodes,
                               embedding_dim=prep_len,
                               # pre_trained=features,
                               # output_dim=prep_len
                               )
        self.edge_prep = nn.ModuleList([nn.Linear(problem.edge_dim, prep_len, bias=False) for _ in range(self.n_mp)])
        self.input_dim = self.prep.output_dim

        # Network
        for mp in range(self.n_mp):
            # agg_layers = []
            # edge_layers = []
            input_dim = self.input_dim
            out_dim = 0
            for i in range(len(node_layer_specs)):
                node_spec = node_layer_specs[i]
                edge_spec = edge_layer_specs[i]
                if False:
                    edge = nn.ModuleList([IdEdgeAggregator(
                        input_dim=input_dim,
                        edge_dim=self.edge_dim,
                        activation=spec['activation'],
                        dropout=self.dropout,
                        batchnorm=self.batchnorm,
                    ) for _ in range(n_head)])
                else:
                    edge_agg = nn.ModuleList([edge_aggr_class(
                        input_dim=input_dim,
                        edge_dim=input_dim,
                        activation=edge_spec['activation'],
                        output_dim=edge_spec['output_dim'],
                        concat_node=edge_spec['concat_node'],
                        concat_edge=edge_spec['concat_edge'],
                        dropout=self.dropout,
                        attn_dropout=self.attn_dropout,
                        batchnorm=self.batchnorm,
                    ) for _ in range(n_head)])
                node_agg = nn.ModuleList([aggregator_class(
                    input_dim=input_dim,
                    edge_dim=edge_spec['output_dim'],
                    output_dim=node_spec['output_dim'],
                    activation=node_spec['activation'],
                    concat_node=node_spec['concat_node'],
                    concat_edge=node_spec['concat_edge'],
                    dropout=self.dropout,
                    attn_dropout=self.attn_dropout,
                    batchnorm=self.batchnorm,
                ) for _ in range(n_head)])
                # agg_layers.append(agg)
                # May not be the same as spec['output_dim']
                input_dim = node_agg[0].output_dim * n_head
                out_dim += input_dim

                # edge_layers.append(edge)
                self.add_module('node_agg_{}_{}'.format(mp, i), node_agg)
                self.add_module('edge_agg_{}_{}'.format(mp, i), edge_agg)
        input_dim = out_dim

        self.mp_agg = mpaggr_class(
            input_dim, n_head=self.n_mp + int(self.bias), dropout=self.dropout, batchnorm=self.batchnorm, )

        self.fc = nn.Sequential(*[
            # nn.Linear(self.mp_agg.output_dim, 32, bias=True),
            # nn.ReLU(), nn.Dropout(self.dropout),
            # nn.Linear(32, problem.n_classes, bias=True),

            nn.Linear(self.mp_agg.output_dim,  problem.n_classes, bias=True),
        ])

    # We only want to forward IDs to facilitate nn.DataParallelism
    def forward(self, train_ids, train=True):

        # print("\tIn Model: input size ", ids.shape)
        # ids.to(self.feats.device)

        has_feats = self.feats is not None
        reg_loss = 0
        output = []
        all_ids = torch.arange(self.n_nodes).to(self.edge_emb_0.device)

        for mp in range(self.n_mp):
            # import GPUtil
            # GPUtil.showUtilization()
            all_feats = self.feats[all_ids].detach() if has_feats else None
            dummy_feats = self.prep(all_ids, all_feats, layer_idx=0)
            all_feats = self.prep(all_ids, all_feats, layer_idx=1)

            all_edges = self.edge_prep[mp](getattr(self, 'edge_emb_{}'.format(mp)))
            # node_neigh = getattr(self, 'node_neigh_{}'.format(mp))  # row: neighbors of a node
            node2edge_idx = getattr(self, 'node2edge_idx_{}'.format(
                mp)) # entries: index of edge embedding, correspondding to node_neigh
            # edge_neigh = getattr(self, 'edge_neigh_{}'.format(mp))  # row: neighbors of a edge
            # edge2node_idx = getattr(self, 'edge2node_idx_{}'.format(mp))
            edge_node_adj = getattr(self, 'edge_node_adj_{}'.format(mp))
            skip_buffer = []
            for layer_idx in range(self.depth):
                # ---Update edges---
                tmp_edges = []
                edge_ids = torch.arange(all_edges.shape[0]).to(self.edge_emb_0.device).detach()
                for chunk_id, chunk in enumerate(torch.split(edge_ids, self.train_batch, dim=0)):
                    chunk_feat = all_edges[chunk]
                    neigh_feat = None
                    chunk_node = all_feats[edge_node_adj[chunk]].view(-1,all_feats.shape[1])

                    chunk_result = torch.cat([getattr(self, 'edge_agg_{}_{}'.format(mp, layer_idx))[h] \
                                                  (chunk_feat, chunk_node, neigh_feat, mask=None) \
                                              for h in range(self.n_head)], dim=1)
                    chunk_result = F.dropout(chunk_result, self.dropout, training=self.training)
                    # del neigh_feat
                    # del chunk_node
                    # del chunk_feat
                    tmp_edges.append(chunk_result)
                    # del chunk_result
                next_edges = torch.cat(tmp_edges, dim=0)
                # ---Update nodes---
                # Split all_ids into batches, in case of OOM.
                tmp_feats = []
                for chunk_id, chunk in enumerate(torch.split(all_ids, self.train_batch, dim=0)):
                    chunk_feat = all_feats[chunk] if layer_idx != 0 else dummy_feats[chunk]
                    neigh_feat = None
                    chunk_edge = all_edges[node2edge_idx[chunk]].view(-1,all_edges.shape[1])

                    chunk_result = torch.cat([getattr(self, 'node_agg_{}_{}'.format(mp, layer_idx))[h] \
                                                  (chunk_feat, chunk_edge,
                                                   neigh_feat, mask=None) \
                                              for h in range(self.n_head)], dim=1)
                    chunk_result = F.dropout(chunk_result, self.dropout, training=self.training)
                    # del neigh_feat
                    # del chunk_edge
                    # del chunk_feat
                    tmp_feats.append(chunk_result)
                    # del chunk_result
                    pass
                next_feats = torch.cat(tmp_feats, dim=0)
                skip_buffer.append(next_feats)
                all_feats = next_feats
                all_edges = next_edges


            #-------cos-------
            cos_all_feats = self.feats[all_ids].detach() if has_feats else None
            cos_dummy_feats = self.prep(all_ids, cos_all_feats, layer_idx=0)
            cos_all_feats = self.prep(all_ids, cos_all_feats, layer_idx=1)

            cos_all_edges = self.edge_prep[mp](getattr(self, 'cos_edge_emb_{}'.format(mp)))
            # node_neigh = getattr(self, 'node_neigh_{}'.format(mp))  # row: neighbors of a node
            cos_node2edge_idx = getattr(self, 'cos_node2edge_idx_{}'.format(
                mp)) # entries: index of edge embedding, correspondding to node_neigh
            # edge_neigh = getattr(self, 'edge_neigh_{}'.format(mp))  # row: neighbors of a edge
            # edge2node_idx = getattr(self, 'edge2node_idx_{}'.format(mp))
            cos_edge_node_adj = getattr(self, 'cos_edge_node_adj_{}'.format(mp))
            cos_skip_buffer = []
            for layer_idx in range(self.depth):
                # ---Update edges---
                cos_tmp_edges = []
                cos_edge_ids = torch.arange(cos_all_edges.shape[0]).to(self.cos_edge_emb_0.device).detach()
                for chunk_id, chunk in enumerate(torch.split(cos_edge_ids, self.train_batch, dim=0)):
                    chunk_feat = cos_all_edges[chunk]
                    neigh_feat = None
                    chunk_node = cos_all_feats[cos_edge_node_adj[chunk]].view(-1,cos_all_feats.shape[1])

                    chunk_result = torch.cat([getattr(self, 'edge_agg_{}_{}'.format(mp, layer_idx))[h] \
                                                  (chunk_feat, chunk_node, neigh_feat, mask=None) \
                                              for h in range(self.n_head)], dim=1)
                    chunk_result = F.dropout(chunk_result, self.dropout, training=self.training)
                    # del neigh_feat
                    # del chunk_node
                    # del chunk_feat
                    cos_tmp_edges.append(chunk_result)
                    # del chunk_result
                cos_next_edges = torch.cat(cos_tmp_edges, dim=0)
                # ---Update nodes---
                # Split all_ids into batches, in case of OOM.
                cos_tmp_feats = []
                for chunk_id, chunk in enumerate(torch.split(all_ids, self.train_batch, dim=0)):
                    chunk_feat = cos_all_feats[chunk] if layer_idx != 0 else cos_dummy_feats[chunk]
                    neigh_feat = None
                    chunk_edge = cos_all_edges[cos_node2edge_idx[chunk]].view(-1,cos_all_edges.shape[1])

                    chunk_result = torch.cat([getattr(self, 'node_agg_{}_{}'.format(mp, layer_idx))[h] \
                                                  (chunk_feat, chunk_edge,
                                                   neigh_feat, mask=None) \
                                              for h in range(self.n_head)], dim=1)
                    chunk_result = F.dropout(chunk_result, self.dropout, training=self.training)
                    # del neigh_feat
                    # del chunk_edge
                    # del chunk_feat
                    cos_tmp_feats.append(chunk_result)
                    # del chunk_result
                    pass
                cos_next_feats = torch.cat(cos_tmp_feats, dim=0)
                cos_skip_buffer.append(cos_next_feats)
                cos_all_feats = cos_next_feats
                cos_all_edges = cos_next_edges
            # del all_feats
            # del all_edges

            reg_loss += ((torch.cat(skip_buffer, dim=-1)-torch.cat(cos_skip_buffer, dim=-1))**2).sum()

            # Jumping connections
            output.append(torch.cat(skip_buffer, dim=-1)[train_ids].unsqueeze(
                0))  # concat skip connections; unsqueeze for metapath aggr.

        output = torch.cat(output, dim=0)

        # output = F.normalize(output, dim=2) #normalize before attention
        # import GPUtil
        # GPUtil.showUtilization()
        output, weights = self.mp_agg(output)
        # output = torch.sum(output,dim=0).squeeze()
        # weights = None
        # print(weights)
        # output = F.normalize(output, dim=-1)  # ?? Do we actually want this? ... Sometimes ...
        output = F.dropout(output, self.dropout, training=self.training)
        output = self.fc(output)
        # reg_loss *= self.coff
        return output, weights, reg_loss
