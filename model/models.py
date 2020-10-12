import torch
import torch.nn as nn
from torch.nn import functional as F
from model.layers import BaseConch,BaseConchNc,BaseConchRd, AvgReadout, Discriminator

class conch_dgi(nn.Module):
    def __init__(self, n_mp,
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
                 bias=False,):
        super(conch_dgi, self).__init__()
        self.dropout = dropout
        self.gcn = BaseConch(
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
                 attn_dropout,
                 bias)

        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(self.gcn.output_dim)
        
        self.mp_agg = mpaggr_class(
            self.gcn.output_dim, n_head=n_mp + int(bias), dropout=dropout, batchnorm=batchnorm, )
        
        self.fc = nn.Sequential(*[
            # nn.Linear(self.mp_agg.output_dim, 32, bias=True),
            # nn.ReLU(), nn.Dropout(self.dropout),
            # nn.Linear(32, problem.n_classes, bias=True),

            nn.Linear(self.mp_agg.output_dim, problem.n_classes, bias=True),
        ])

    def forward(self, feat1, feat2, msk, samp_bias1, samp_bias2, get_embed=False):
        h_1 = self.gcn(feat1)
        
        # h_1 = F.normalize(h_1, dim=2) #normalize before attention
        output, weights = self.mp_agg(h_1)
        output = self.fc(output)
        preds = F.dropout(output, self.dropout, training=self.training)
        if get_embed:
            return preds

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = self.gcn(feat2)

        reg = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return preds, weights, reg
    
    def get_embed(self, feat1):
        h_1 = self.gcn(feat1)
        output, _ = self.mp_agg(h_1)
        output = self.fc(output)

        return output


class conch_dgi2(nn.Module):
    def __init__(self, n_mp,
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
                 shuffle=False,
                 attn_dropout=0,
                 bias=False,):
        super(conch_dgi2, self).__init__()
        self.dropout = dropout
        self.shuffle = shuffle
        self.gcn = BaseConch(
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
                 attn_dropout,
                 bias)

        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        
        
        self.mp_agg = mpaggr_class(
            self.gcn.output_dim, n_head=n_mp + int(bias), dropout=dropout, batchnorm=batchnorm, )
        
        self.disc = Discriminator(self.mp_agg.output_dim)
        
        self.fc = nn.Sequential(*[
            # nn.Linear(self.mp_agg.output_dim, 32, bias=True),
            # nn.ReLU(), nn.Dropout(self.dropout),
            # nn.Linear(32, problem.n_classes, bias=True),

            nn.Linear(self.mp_agg.output_dim, problem.n_classes, bias=True),
        ])

    def forward(self, feat1, feat2, msk, samp_bias1, samp_bias2, get_embed=None):
        h_1 = self.gcn(feat1,shuffle=False)
        
        # h_1 = F.normalize(h_1, dim=2) #normalize before attention
        h_1, weights = self.mp_agg(h_1)
        output = self.fc(h_1)
        preds = F.dropout(output, self.dropout, training=self.training)
        if get_embed=='embed':
            return h_1
        if get_embed=='pred':
            return output

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = self.gcn(feat2, shuffle=self.shuffle)
        h_2, _ = self.mp_agg(h_2)

        reg = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return preds, weights, reg
    
    def get_embed(self, feat1):
        h_1 = self.gcn(feat1)
        output, _ = self.mp_agg(h_1)
        output = self.fc(output)

        return output


class conch_nc(nn.Module):
    def __init__(self, n_mp,
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
                 shuffle=False,
                 attn_dropout=0,
                 bias=False,):
        super(conch_nc, self).__init__()
        self.dropout = dropout
        self.gcn = BaseConchNc(
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
                 attn_dropout,
                 bias)

        # self.read = AvgReadout()

        # self.sigm = nn.Sigmoid()

        # self.disc = Discriminator(self.gcn.output_dim)
        
        self.mp_agg = mpaggr_class(
            self.gcn.output_dim, n_head=n_mp + int(bias), dropout=dropout, batchnorm=batchnorm, )
        
        self.fc = nn.Sequential(*[
            # nn.Linear(self.mp_agg.output_dim, 32, bias=True),
            # nn.ReLU(), nn.Dropout(self.dropout),
            # nn.Linear(32, problem.n_classes, bias=True),

            nn.Linear(self.mp_agg.output_dim, problem.n_classes, bias=True),
        ])
        self.shuffle = shuffle

    def forward(self, feat1, feat2, msk, samp_bias1, samp_bias2, get_embed=None):
        h_1 = self.gcn(feat1)
        
        # h_1 = F.normalize(h_1, dim=2) #normalize before attention
        h_1, weights = self.mp_agg(h_1)
        output = self.fc(h_1)
        preds = F.dropout(output, self.dropout, training=self.training)

        if get_embed=='embed':
            return h_1
        if get_embed=='pred':
            return output

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = self.gcn(feat2)
        h_2, _ = self.mp_agg(h_2)

        reg = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return preds, weights, reg
    
    def get_embed(self, feat1):
        h_1 = self.gcn(feat1)
        h_1, weights = self.mp_agg(h_1)
        # output = self.fc(output)

        return h_1

    def get_predict(self, feat1):
        h_1 = self.gcn(feat1)
        h_1, weights = self.mp_agg(h_1)
        output = self.fc(output)

        return output



class conch_rd(nn.Module):
    def __init__(self, n_mp,
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
                 shuffle = False,
                 attn_dropout=0,
                 bias=False,):
        super(conch_rd, self).__init__()
        self.dropout = dropout
        self.gcn = BaseConchRd(
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
                 attn_dropout,
                 bias)

        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(self.gcn.output_dim)
        
        self.mp_agg = mpaggr_class(
            self.gcn.output_dim, n_head=n_mp + int(bias), dropout=dropout, batchnorm=batchnorm, )
        
        self.fc = nn.Sequential(*[
            # nn.Linear(self.mp_agg.output_dim, 32, bias=True),
            # nn.ReLU(), nn.Dropout(self.dropout),
            # nn.Linear(32, problem.n_classes, bias=True),

            nn.Linear(self.mp_agg.output_dim, problem.n_classes, bias=True),
        ])
        self.shuffle = shuffle

    def forward(self, feat1, feat2, msk, samp_bias1, samp_bias2, get_embed=None):
        h_1 = self.gcn(feat1,shuffle=False)
        
        # h_1 = F.normalize(h_1, dim=2) #normalize before attention
        h_1, weights = self.mp_agg(h_1)
        output = self.fc(h_1)
        preds = F.dropout(output, self.dropout, training=self.training)
        if get_embed:
            return preds

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = self.gcn(feat2,shuffle=self.shuffle)
        h_2, _ = self.mp_agg(h_2)

        reg = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return preds, weights, reg
    
    def get_embed(self, feat1):
        h_1 = self.gcn(feat1)
        output, _ = self.mp_agg(h_1)
        output = self.fc(output)

        return output