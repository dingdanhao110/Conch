from __future__ import division
from __future__ import print_function

from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from lr import LRSchedule

from nn_modules import GraphConvolution, IdEdgeAggregator
import numpy as np

class BipartiteGCNRandom(nn.Module):
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

        super(BipartiteGCNRandom, self).__init__()

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
        
        from time import time
        #First random sample sub graph.
        start = time()
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


        end = time()
        print('sample time:',end-start)
        has_feats = self.feats is not None

        output = []
        all_ids = torch.arange(self.n_nodes).to(self.edge_emb_0.device)

        copy_time=0

        for mp in range(self.n_mp):
            # import GPUtil
            # GPUtil.showUtilization()
            all_feats = self.feats[all_ids].detach() if has_feats else None
            dummy_feats = self.prep(all_ids, all_feats, layer_idx=0)
            all_feats = self.prep(all_ids, all_feats, layer_idx=1)

            all_edges = self.edge_prep[mp](getattr(self, 'edge_emb_{}'.format(mp)))#[edges_to_update_arr[mp]]
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
                start = time()
                next_edges = torch.cuda.FloatTensor(all_edges.shape[0],edge_out.shape[1])
                next_edges[edge_ids] = edge_out
                end = time()
                copy_time += end-start
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
            output.append(torch.cat(skip_buffer, dim=-1)[train_ids].unsqueeze(
                0))  # concat skip connections; unsqueeze for metapath aggr.

        output = torch.cat(output, dim=0)
        print('copy time:',copy_time)
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
        return output, weights
