from __future__ import division
from __future__ import print_function

from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from lr import LRSchedule
import numpy as np
from nn_modules import IdEdgeAggregator

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        # print(c.shape) # B,D
        # print(h_pl.shape)
        # print(h_mi.shape)
        
        D = h_pl.shape[-1]

        if (len(h_pl.shape)==2):
            h_pl = h_pl.view(1,-1,D)
            h_mi = h_mi.view(1,-1,D)
        
        B = h_pl.shape[0] #B=1 for 2d input
        N = h_pl.shape[1]

        c_x = c.view(B,1,D)
        c_x = c_x.expand_as(h_pl)
        # print(c_x.shape)

        h_pl = h_pl.view(-1, D)
        c_x = c_x.reshape(-1, D)
        h_mi = h_mi.view(-1, D)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x)).view(B,-1) #B*N
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x)).view(B,-1)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1) #B*2N
        # print(logits.shape)
        return logits

class BaseConch(nn.Module):
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
                 attn_dropout=0,
                 bias=False,
                 ):

        super(BaseConch, self).__init__()

        # --
        # Input Data
        self.edge_dim = problem.edge_dim
        self.input_dim = problem.feats_dim
        self.n_nodes = problem.n_nodes
        self.n_classes = problem.n_classes
        self.n_head = n_head
        self.bias = bias

        # feat is dynamically inputted 
        # # self.feats
        # self.register_buffer('feats', problem.feats)

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

        # Define network
        self.n_mp = n_mp
        self.depth = len(node_layer_specs)
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.batchnorm = batchnorm

        # No more samplers

        self.train_batch = 999999

        # features = torch.FloatTensor(features[:problem.n_nodes,:])
        print("feat dim: {}, prep_len: {}".format(problem.feats_dim,prep_len))
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

        self.output_dim = out_dim 

    # We only want to forward IDs to facilitate nn.DataParallelism
    def forward(self, feats, shuffle=False):

        # ids.to(self.feats.device)
        has_feats = feats is not None
        
        output = []
        all_ids = torch.arange(self.n_nodes,device=self.edge_emb_0.device)#.cuda()#
        # print(all_ids.device, feats.device)

        for mp in range(self.n_mp):
            if shuffle:
                edge_perm = torch.randperm(getattr(self, 'edge_emb_{}'.format(mp)).shape[0],device=self.edge_emb_0.device)#.cuda()#
            else:
                edge_perm = torch.arange(getattr(self, 'edge_emb_{}'.format(mp)).shape[0],device=self.edge_emb_0.device)#.cuda()#

            # import GPUtil
            # GPUtil.showUtilization()
            all_feats = feats[all_ids].detach() if has_feats else None
            dummy_feats = self.prep(all_ids, all_feats, layer_idx=0)
            all_feats = self.prep(all_ids, all_feats, layer_idx=1)

            all_edges = self.edge_prep[mp](getattr(self, 'edge_emb_{}'.format(mp)))[edge_perm]
            # print(all_edges.shape)
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
                # print(all_edges.shape)

                edge_ids = torch.arange(all_edges.shape[0],device=self.edge_emb_0.device) #.cuda()
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

            # Jumping connections
            output.append(torch.cat(skip_buffer, dim=-1).unsqueeze(
                0))  # concat skip connections; unsqueeze for metapath aggr.

        output = torch.cat(output, dim=0)
        
        return output


class BaseConchNc(nn.Module):
    """
    No context.
    Node feature: [N,d]
    Node neighbors: [N,S]
    No Edge embeddings
    Node edge mapping: [N,S]->E
    Edge neighbors: [E,2S]
    """

    def __init__(self,
                 n_mp,
                 problem,
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
                 attn_dropout=0,
                 bias=False,
                 ):

        super(BaseConchNc, self).__init__()

        # --
        # Input Data
        self.edge_dim = problem.edge_dim
        self.input_dim = problem.feats_dim
        self.n_nodes = problem.n_nodes
        self.n_classes = problem.n_classes
        self.n_head = n_head
        self.bias = bias

        # self.feats
        # self.register_buffer('feats', problem.feats)

        # self.edge_neigh_mp
        # for i, key in enumerate(problem.edge_neighs):
        #     self.register_buffer('edge_neigh_{}'.format(i),
        #                          problem.edge_neighs[key])

        # self.node_neigh_mp
        for i, key in enumerate(problem.node_neighs):
            self.register_buffer('node_neigh_{}'.format(i),
                                 problem.node_neighs[key])

        # self.node2edge_idx_mp
        # for i, key in enumerate(problem.node2edge_idxs):
        #     self.register_buffer('node2edge_idx_{}'.format(i),
        #                          problem.node2edge_idxs[key])

        # # self.edge_emb_mp
        # for i, key in enumerate(problem.edge_embs):
        #     self.register_buffer('edge_emb_{}'.format(i),
        #                          problem.edge_embs[key])
        #     print(problem.edge_embs[key].shape)

        # # self.edge2node_idx_mp
        # for i, key in enumerate(problem.edge2node_idxs):
        #     self.register_buffer('edge2node_idx_{}'.format(i),
        #                          problem.edge2node_idxs[key])

        # self.edge_node_adj_mp
        # for i, key in enumerate(problem.edge_node_adjs):
        #     self.register_buffer('edge_node_adj_{}'.format(i),
        #                          problem.edge_node_adjs[key])

        # Define network
        self.n_mp = n_mp
        self.depth = len(node_layer_specs)
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.batchnorm = batchnorm

        # No more samplers

        self.train_batch = 9999999

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
                node_agg = nn.ModuleList([aggregator_class(
                    input_dim=input_dim,
                    edge_dim=0,
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
                
        self.output_dim = out_dim

        # self.mp_agg = mpaggr_class(
        #     input_dim, n_head=self.n_mp + int(self.bias), dropout=self.dropout, batchnorm=self.batchnorm, )

        # self.fc = nn.Sequential(*[
        #     # nn.Linear(self.mp_agg.output_dim, 32, bias=True),
        #     # nn.ReLU(), nn.Dropout(self.dropout),
        #     # nn.Linear(32, problem.n_classes, bias=True),

        #     nn.Linear(self.mp_agg.output_dim,  problem.n_classes, bias=True),
        # ])

    # We only want to forward IDs to facilitate nn.DataParallelism
    def forward(self, feats):

        # print("\tIn Model: input size ", ids.shape)
        # ids.to(self.feats.device)

        has_feats = feats is not None

        output = []
        all_ids = torch.arange(self.n_nodes).to(self.node_neigh_0.device)
        # print( getattr(self, 'node_neigh_{}'.format(0)).requires_grad)
        for mp in range(self.n_mp):
            # import GPUtil
            # GPUtil.showUtilization()
            all_feats = feats[all_ids].detach()  if has_feats else None
            dummy_feats = self.prep(all_ids, all_feats, layer_idx=0)
            all_feats = self.prep(all_ids, all_feats, layer_idx=1)

            # all_edges = self.edge_prep[mp](getattr(self, 'edge_emb_{}'.format(mp)))
            node_neigh = getattr(self, 'node_neigh_{}'.format(mp))  # row: neighbors of a node
            # node2edge_idx = getattr(self, 'node2edge_idx_{}'.format(
            #     mp)) # entries: index of edge embedding, correspondding to node_neigh
            # edge_neigh = getattr(self, 'edge_neigh_{}'.format(mp))  # row: neighbors of a edge
            # edge2node_idx = getattr(self, 'edge2node_idx_{}'.format(mp))
            # edge_node_adj = getattr(self, 'edge_node_adj_{}'.format(mp))
            skip_buffer = []
            for layer_idx in range(self.depth):
                
                # ---Update nodes---
                # Split all_ids into batches, in case of OOM.
                tmp_feats = []
                for chunk_id, chunk in enumerate(torch.split(all_ids, self.train_batch, dim=0)):
                    chunk_feat = all_feats[chunk] if layer_idx != 0 else dummy_feats[chunk]
                    neigh_feat = None
                    chunk_edge = all_feats[node_neigh[chunk]].view(-1,all_feats.shape[1])

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
            # Jumping connections
            output.append(torch.cat(skip_buffer, dim=-1).unsqueeze(
                0))  # concat skip connections; unsqueeze for metapath aggr.

        output = torch.cat(output, dim=0)

        # # output = F.normalize(output, dim=2) #normalize before attention
        # # import GPUtil
        # # GPUtil.showUtilization()
        # output, weights = self.mp_agg(output)
        # # output = torch.sum(output,dim=0).squeeze()
        # # weights = None
        # # print(weights)
        # # output = F.normalize(output, dim=-1)  # ?? Do we actually want this? ... Sometimes ...
        # output = F.dropout(output, self.dropout, training=self.training)
        # output = self.fc(output)
        return output

class BaseConchRd(nn.Module):
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
                 K,
                 attn_dropout=0,
                 bias=False,
                 ):

        super(BaseConchRd, self).__init__()

        # --
        # Input Data
        self.edge_dim = problem.edge_dim
        self.input_dim = problem.feats_dim
        self.n_nodes = problem.n_nodes
        self.n_classes = problem.n_classes
        self.n_head = n_head
        self.bias = bias
        self.K = K
        # self.feats
        # self.register_buffer('feats', problem.feats)

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
            self.register_buffer('Node2edge_idx_{}'.format(i),
                                 problem.node2edge_idxs[key])

        # self.edge_emb_mp
        for i, key in enumerate(problem.edge_embs):
            self.register_buffer('edge_emb_{}'.format(i),
                                 problem.edge_embs[key])
            print(problem.edge_embs[key].shape)

        # # self.edge2node_idx_mp
        # for i, key in enumerate(problem.edge2node_idxs):
        #     self.register_buffer('edge2node_idx_{}'.format(i),
        #                          problem.edge2node_idxs[key])

        # self.edge_node_adj_mp
        for i, key in enumerate(problem.edge_node_adjs):
            self.register_buffer('edge_node_adj_{}'.format(i),
                                 problem.edge_node_adjs[key])

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
        self.output_dim = out_dim 

        # self.mp_agg = mpaggr_class(
        #     input_dim, n_head=self.n_mp + int(self.bias), dropout=self.dropout, batchnorm=self.batchnorm, )

        # self.fc = nn.Sequential(*[
        #     # nn.Linear(self.mp_agg.output_dim, 32, bias=True),
        #     # nn.ReLU(), nn.Dropout(self.dropout),
        #     # nn.Linear(32, problem.n_classes, bias=True),

        #     nn.Linear(self.mp_agg.output_dim,  problem.n_classes, bias=True),
        # ])

    # We only want to forward IDs to facilitate nn.DataParallelism
    def forward(self, feats, shuffle=False):
        
        # from time import time
        #First random sample sub graph.
        # start = time()
        node2edge_idx_arr=[]
        edges_to_update_arr=[]
        node2edge_idx_arr_new=[]
        edges_to_update_arr_new=[]
        for mp in range(self.n_mp):
            
            sel = np.random.choice(self.Node2edge_idx_0.shape[1], (self.Node2edge_idx_0.shape[0], self.K))
            node2edge_idx_arr.append ( getattr(self, 'Node2edge_idx_{}'.format(
                mp))[
                np.arange(self.Node2edge_idx_0.shape[0]).repeat(self.K).reshape(-1),
                np.array(sel).reshape(-1)
                ] )
            node2edge_idx_arr[mp] = node2edge_idx_arr[mp].view(-1,self.K)
            #recalcualte edges to update:
            edges_to_update_arr.append(torch.unique(node2edge_idx_arr[mp]).view(-1))
            
            #remap edge to new edge id
            # mapper=dict()
            # for i in edges_to_update_arr[mp]:
            #     x=i.item()
            #     mapper[x] = len(mapper)
            
            # node2edge_idx_arr_new.append( torch.LongTensor([mapper[i.item()] for i in node2edge_idx_arr[mp] ]).view(-1,self.K).cuda())
            # edges_to_update_arr_new.append(torch.LongTensor([mapper[i.item()] for i in edges_to_update_arr[mp] ]).cuda())


        # end = time()
        # print('sample time:',end-start)
        has_feats = feats is not None

        output = []
        all_ids = torch.arange(self.n_nodes).to(self.edge_emb_0.device)

        # copy_time=0

        for mp in range(self.n_mp):
            if shuffle:
                edge_perm = torch.randperm(getattr(self, 'edge_emb_{}'.format(mp)).shape[0]).cuda()#,device=self.edge_emb_0.device
            else:
                edge_perm = torch.arange(getattr(self, 'edge_emb_{}'.format(mp)).shape[0]).cuda()#,device=self.edge_emb_0.device

            # import GPUtil
            # GPUtil.showUtilization()
            all_feats = feats[all_ids].detach() if has_feats else None
            dummy_feats = self.prep(all_ids, all_feats, layer_idx=0)
            all_feats = self.prep(all_ids, all_feats, layer_idx=1)

            all_edges = self.edge_prep[mp](getattr(self, 'edge_emb_{}'.format(mp)))[edge_perm]#[edges_to_update_arr[mp]]
            # node_neigh = getattr(self, 'node_neigh_{}'.format(mp))  # row: neighbors of a node
            node2edge_idx = node2edge_idx_arr[mp] # entries: index of edge embedding, correspondding to node_neigh
            # edge_neigh = getattr(self, 'edge_neigh_{}'.format(mp))  # row: neighbors of a edge
            # edge2node_idx = getattr(self, 'edge2node_idx_{}'.format(mp))
            edge_node_adj = getattr(self, 'edge_node_adj_{}'.format(mp))
            skip_buffer = []
            for layer_idx in range(self.depth):
                # ---Update edges---
                tmp_edges = []
                # edge_ids = torch.arange(all_edges.shape[0]).to(self.edge_emb_0.device).detach()
                edge_ids = edges_to_update_arr[mp]
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
                # next_edges = torch.cat(tmp_edges, dim=0)
                edge_out = torch.cat(tmp_edges, dim=0)
                # start = time()
                next_edges = torch.cuda.FloatTensor(all_edges.shape[0],edge_out.shape[1])
                next_edges[edge_ids] = edge_out
                # end = time()
                # copy_time += end-start
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

            del all_feats
            del all_edges

            # Jumping connections
            output.append(torch.cat(skip_buffer, dim=-1).unsqueeze(
                0))  # concat skip connections; unsqueeze for metapath aggr.

        output = torch.cat(output, dim=0)
        # print('copy time:',copy_time)
        # # output = F.normalize(output, dim=2) #normalize before attention
        # # import GPUtil
        # # GPUtil.showUtilization()
        # output, weights = self.mp_agg(output)
        # # output = torch.sum(output,dim=0).squeeze()
        # # weights = None
        # # print(weights)
        # # output = F.normalize(output, dim=-1)  # ?? Do we actually want this? ... Sometimes ...
        # output = F.dropout(output, self.dropout, training=self.training)
        # output = self.fc(output)
        return output


class BaseConchGS(nn.Module):
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
                 attn_dropout=0,
                 bias=False,
                 ):

        super(BaseConchGS, self).__init__()

        # --
        # Input Data
        self.edge_dim = problem.edge_dim
        self.input_dim = problem.feats_dim
        self.n_nodes = problem.n_nodes
        self.n_classes = problem.n_classes
        self.n_head = n_head
        self.bias = bias

        # feat is dynamically inputted 
        # # self.feats
        # self.register_buffer('feats', problem.feats)

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

        # Define network
        self.n_mp = n_mp
        self.depth = len(node_layer_specs)
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.batchnorm = batchnorm

        # features = torch.FloatTensor(features[:problem.n_nodes,:])
        print("feat dim: {}, prep_len: {}".format(problem.feats_dim,prep_len))
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

        self.output_dim = out_dim 

    # feats correponds to all nodes; not prep()ed
    def forward(self, ids, feats, shuffle=False):

        ids.to(feats.device)
        # has_feats = feats is not None
        
        output = []
        # all_ids = torch.arange(self.n_nodes,device=self.edge_emb_0.device)#.cuda()#
        # print(all_ids.device, feats.device)
        

        for mp in range(self.n_mp):
            if shuffle:
                edge_perm = torch.randperm(getattr(self, 'edge_emb_{}'.format(mp)).shape[0],device=self.edge_emb_0.device)#.cuda()#
            else:
                edge_perm = torch.arange(getattr(self, 'edge_emb_{}'.format(mp)).shape[0],device=self.edge_emb_0.device)#.cuda()#

            tmp_ids = ids
            
            edge_emb = self.edge_prep[mp](getattr(self, 'edge_emb_{}'.format(mp)))[edge_perm]
            # print(all_edges.shape)
            # node_neigh = getattr(self, 'node_neigh_{}'.format(mp))  # row: neighbors of a node
            node2edge_idx = getattr(self, 'node2edge_idx_{}'.format(
                mp)) # entries: index of edge embedding, correspondding to node_neigh
            # edge_neigh = getattr(self, 'edge_neigh_{}'.format(mp))  # row: neighbors of a edge
            # edge2node_idx = getattr(self, 'edge2node_idx_{}'.format(mp))
            edge_node_adj = getattr(self, 'edge_node_adj_{}'.format(mp))

            # all_feats = feats[tmp_ids] if has_feats else None
            # dummy_feats = self.prep(tmp_ids, all_feats, layer_idx=0)
            all_feats = [self.prep(tmp_ids, feats[tmp_ids], layer_idx=0)]
            # all_feats = [feats[tmp_ids]]

            # ===============Sampling!!!!====================

            for layer_idx in range(self.depth):
                if layer_idx % 2 ==0:
                    # node->edge
                    # next_adj = node2edge_idx
                    tmp_ids = node2edge_idx[tmp_ids].contiguous().view(-1)
                    next_feat = edge_emb[tmp_ids]
                    all_feats.append(next_feat)
                else:
                    # edge->node
                    # next_adj = edge_node_adj
                    tmp_ids = edge_node_adj[tmp_ids].contiguous().view(-1)
                    next_feat = self.prep(tmp_ids, feats[tmp_ids], layer_idx=layer_idx+1)
                    all_feats.append(next_feat)  # or else; .view(-1,all_feats.shape[1])
                # k = next_adj.shape[1]
                # print(tmp_ids.shape)

            # ================End of Sampling ==================

            skip_buffer = []
            for layer_idx in range(self.depth):
                
                all_feats = [ getattr(self, 'node_agg_{}_{}'.format(mp, layer_idx))[0] \
                                                  (all_feats[k], all_feats[k + 1],
                                                   None, mask=None)\
                              if k%2==0 else \
                          getattr(self, 'edge_agg_{}_{}'.format(mp, layer_idx))[0] \
                                                  (all_feats[k], all_feats[k + 1], None, mask=None)
                          for k in range(len(all_feats)-1) ]
                all_feats = [
                    F.dropout(i, self.dropout, training=self.training) for i in all_feats]
                skip_buffer.append(all_feats[0])
            assert len(all_feats) == 1, "len(all_feats) != 1"
                # # ---Update edges---
                # tmp_edges = []
                # # print(all_edges.shape)

                # edge_ids = torch.arange(all_edges.shape[0],device=self.edge_emb_0.device) #.cuda()
                # for chunk_id, chunk in enumerate(torch.split(edge_ids, self.train_batch, dim=0)):
                #     chunk_feat = all_edges[chunk]
                #     neigh_feat = None
                #     chunk_node = all_feats[edge_node_adj[chunk]].view(-1,all_feats.shape[1])

                #     chunk_result = torch.cat([getattr(self, 'edge_agg_{}_{}'.format(mp, layer_idx))[h] \
                #                                   (chunk_feat, chunk_node, neigh_feat, mask=None) \
                #                               for h in range(self.n_head)], dim=1)
                #     chunk_result = F.dropout(chunk_result, self.dropout, training=self.training)
                #     # del neigh_feat
                #     # del chunk_node
                #     # del chunk_feat
                #     tmp_edges.append(chunk_result)
                #     # del chunk_result
                # next_edges = torch.cat(tmp_edges, dim=0)
                # # ---Update nodes---
                # # Split all_ids into batches, in case of OOM.
                # tmp_feats = []
                # for chunk_id, chunk in enumerate(torch.split(all_ids, self.train_batch, dim=0)):
                #     chunk_feat = all_feats[chunk] if layer_idx != 0 else dummy_feats[chunk]
                #     neigh_feat = None
                #     chunk_edge = all_edges[node2edge_idx[chunk]].view(-1,all_edges.shape[1])

                #     chunk_result = torch.cat([getattr(self, 'node_agg_{}_{}'.format(mp, layer_idx))[h] \
                #                                   (chunk_feat, chunk_edge,
                #                                    neigh_feat, mask=None) \
                #                               for h in range(self.n_head)], dim=1)
                #     chunk_result = F.dropout(chunk_result, self.dropout, training=self.training)
                #     # del neigh_feat
                #     # del chunk_edge
                #     # del chunk_feat
                #     tmp_feats.append(chunk_result)
                #     # del chunk_result
                #     pass
                # next_feats = torch.cat(tmp_feats, dim=0)
                # skip_buffer.append(next_feats)
                # all_feats = next_feats
                # all_edges = next_edges

            # Jumping connections
            output.append(torch.cat(skip_buffer, dim=-1).unsqueeze(
                0))  # concat skip connections; unsqueeze for metapath aggr.

        output = torch.cat(output, dim=0)
        
        return output

# Applies an average on seq, of shape (Batch, nodes, features) or (Node, Feat)
# While taking into account the masking of msk
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, -2)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)


class CLING_HAN(nn.Module):
    def __init__(self,
                 n_mp,
                 problem,
                 prep_len,
                 n_head,
                 layer_specs,
                 edge_layer_specs,
                 aggregator_class,
                 mpaggr_class,
                 edgeupt_class,
                 prep_class,
                 sampler_class,
                 dropout,
                 batchnorm,
                 attn_dropout=0,
                 bias=False,
                 ):

        super(CLING_HAN, self).__init__()

        # --
        # Input Data
        # self.edge_dim = problem.edge_dim
        self.input_dim = problem.feats_dim
        self.n_nodes = problem.n_nodes
        self.n_classes = problem.n_classes
        self.n_head = n_head

        # self.feats
        # self.register_buffer('feats', problem.feats)

        for i, key in enumerate(problem.node_neighs):
            self.register_buffer('adjs_{}'.format(i), problem.node_neighs[key])

        # Define network
        self.n_mp = n_mp
        self.depth = len(layer_specs)
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.batchnorm = batchnorm
        self.concat_node = False
        self.input_dropout = 0

        # Sampler
        self.train_sampler = sampler_class()
        self.val_sampler = sampler_class()
        self.train_sample_fns = [partial(
            self.train_sampler, n_samples=s['n_train_samples']) for s in layer_specs]
        self.val_sample_fns = [
            partial(self.val_sampler, n_samples=s['n_val_samples']) for s in layer_specs]

        # Prep
        self.prep = prep_class(input_dim=problem.feats_dim, n_nodes=problem.n_nodes,
                               embedding_dim=prep_len,
                               )
        self.input_dim = self.prep.output_dim

        # Network
        for mp in range(self.n_mp):
            # agg_layers = []
            # edge_layers = []
            input_dim = self.input_dim
            out_dim = 0
            for i, spec in enumerate(layer_specs):
                agg = nn.ModuleList([aggregator_class(
                    input_dim=input_dim,
                    edge_dim=None,
                    output_dim=spec['output_dim'],
                    activation=spec['activation'],
                    concat_node=spec['concat_node'],
                    concat_edge=spec['concat_edge'],
                    dropout=self.dropout,
                    attn_dropout=self.attn_dropout,
                    batchnorm=self.batchnorm,
                    hidden_dim=128,
                ) for _ in range(n_head)])
                # agg_layers.append(agg)
                # May not be the same as spec['output_dim']
                input_dim = agg[0].output_dim * n_head
                out_dim = input_dim

                # edge_layers.append(edge)
                self.add_module('agg_{}_{}'.format(mp, i), agg)
        self.output_dim = out_dim


    # We only forward IDs to facilitate nn.DataParallelism
    def forward(self, ids, feats, train=True):

        # Sample neighbors
        sample_fns = self.train_sample_fns if train else self.val_sample_fns

        has_feats = feats is not None

        output = []
        tmp_ids = ids
        tmp_feats = feats[ids] if has_feats else None
        init_feats = torch.nn.functional.dropout( self.prep(ids, tmp_feats, layer_idx=0),self.input_dropout,training=train)
            
        for mp in range(self.n_mp):
            ids = tmp_ids
            #tmp_feats = self.prep(ids, tmp_feats, layer_idx=0)
            tmp_feats = init_feats
            all_feats = [tmp_feats]
            # all_edges = []
            for layer_idx, sampler_fn in enumerate(sample_fns):
                neigh = sampler_fn(
                    getattr(self, 'adjs_{}'.format(mp)), ids=ids)
                # print(neigh.shape, edges.shape)
                # all_edges.append(getattr(self, 'edge_emb_{}'.format(mp))[
                #                  edges.contiguous().view(-1)])

                ids = neigh.contiguous().view(-1)
                tmp_feats = feats[ids] if has_feats else None
                all_feats.append(
                    self.prep(ids, tmp_feats, layer_idx=layer_idx + 1))

            # Sequentially apply layers, per original (little weird, IMO)
            # Each iteration reduces length of array by one
            tmp_out = []
            for i in range(self.depth):
                # all_edges = all_edges

                all_feats = [torch.cat([getattr(self, 'agg_{}_{}'.format(mp, i))[h](
                    all_feats[k], all_feats[k + 1],
                    None, mask=None) for h in range(self.n_head)], dim=1)
                    for k in range(len(all_feats) - 1)]
                all_feats = [
                    F.dropout(i, self.dropout, training=self.training) for i in all_feats]

                tmp_out.append(all_feats[0])
            assert len(all_feats) == 1, "len(all_feats) != 1"
            if self.concat_node:
                output.append(torch.cat(tmp_out, dim=-1).unsqueeze(0))
            else:
                output.append(tmp_out[-1].unsqueeze(0))
        output = torch.cat(output)

        return output