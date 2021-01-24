#!/usr/bin/env python

"""
    helpers.py
"""

from __future__ import print_function

import numpy as np

import torch
from torch.autograd import Variable
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from preprocess.word_emb import count2feat


def set_seeds(seed=0):
    np.random.seed(seed)
    _ = torch.manual_seed(seed)
    if torch.cuda.is_available():
        _ = torch.cuda.manual_seed(seed)


def to_numpy(x):
    if isinstance(x, Variable):
        return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()
    
    return x.cpu().item().item if x.is_cuda else x.item().item


def read_embed(path="./data/dblp/",
               emb_file="APC"):
    with open("{}{}.emb".format(path, emb_file)) as f:
        n_nodes, n_feature = map(int, f.readline().strip().split())
    print("number of nodes:{}, embedding size:{}".format(n_nodes, n_feature))

    embedding = np.loadtxt("{}{}.emb".format(path, emb_file),
                           dtype=np.float32, skiprows=1)
    emb_index = {}
    for i in range(n_nodes):
        emb_index[embedding[i, 0]] = i

    features = np.asarray([embedding[emb_index[i], 1:] for i in range(n_nodes)])

    assert features.shape[1] == n_feature
    assert features.shape[0] == n_nodes

    return features, n_nodes, n_feature

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def load_2hop_index(path="./data/dblp/", file="APA"):
    index = {}
    with open("{}{}.ind".format(path, file), mode='r') as f:
        for line in f:
            array = [int(x) for x in line.split()]
            a1 = array[0]
            a2 = array[1]
            if a1 not in index:
                index[a1] = {}
            if a2 not in index[a1]:
                index[a1][a2] = set()
            for p in array[2:]:
                index[a1][a2].add(p)

    return index


def read_mpindex_dblp(path="./data/dblp2/",train_per=0.1):
    print(train_per)
    label_file = "author_label"
    PA_file = "PA"
    PC_file = "PC"
    PT_file = "PT"
    feat_emb_file = 'term_emb.npy'
    feat_emb = np.load("{}{}".format(path, feat_emb_file))
    # print("{}{}.txt".format(path, PA_file))
    PA = np.genfromtxt("{}{}.txt".format(path, PA_file),
                       dtype=np.int32)
    PC = np.genfromtxt("{}{}.txt".format(path, PC_file),
                       dtype=np.int32)
    PT = np.genfromtxt("{}{}.txt".format(path, PT_file),
                       dtype=np.int32)
    PA[:, 0] -= 1
    PA[:, 1] -= 1
    PC[:, 0] -= 1
    PC[:, 1] -= 1
    PT[:, 0] -= 1
    PT[:, 1] -= 1

    paper_max = max(PA[:, 0]) + 1
    author_max = max(PA[:, 1]) + 1
    term_max = max(PT[:, 1]) + 1

    PA_s = sp.coo_matrix((np.ones(PA.shape[0]), (PA[:, 0], PA[:, 1])),
                         shape=(paper_max, author_max),
                         dtype=np.float32)
    PT_s = sp.coo_matrix((np.ones(PT.shape[0]), (PT[:, 0], PT[:, 1])),
                         shape=(paper_max, term_max),
                         dtype=np.float32)

    # transformer = TfidfTransformer()
    features = PA_s.transpose() * PT_s  # AT
    # features = transformer.fit_transform(features)
    # features = np.array(features.todense())
    features = count2feat(features, feat_emb)

    labels_raw = np.genfromtxt("{}{}.txt".format(path, label_file),
                               dtype=np.int32)
    labels_raw[:, 0] -= 1
    labels_raw[:, 1] -= 1
    labels = np.zeros(author_max)
    labels[labels_raw[:, 0]] = labels_raw[:, 1]

    reordered = np.random.permutation(labels_raw[:, 0])
    total_labeled = labels_raw.shape[0]

    idx_train = reordered[range(int(total_labeled * train_per))]
    idx_val = reordered[range(int(total_labeled * train_per), int(total_labeled * 0.8))]
    idx_test = reordered[range(int(total_labeled * 0.8), total_labeled)]

    folds = {'train':idx_train,'val':idx_val,'test':idx_test}

    return features, labels, folds


def read_mpindex_aminer(path="./data/aminer/",train_per=0.1):
    print(train_per)
    label_file = "label"
    PA_file = "PA"
    PC_file = "PC"
    feat_emb_file = 'paper_fea.npy'
    feat_emb = np.load("{}{}".format(path, feat_emb_file))
    # print("{}{}.txt".format(path, PA_file))
    PA = np.genfromtxt("{}{}.txt".format(path, PA_file),
                       dtype=np.int32)
    PC = np.genfromtxt("{}{}.txt".format(path, PC_file),
                       dtype=np.int32)

    paper_max = max(PA[:, 0]) + 1 #416554
    author_max = max(PA[:, 1]) + 1

    # PA_s = sp.coo_matrix((np.ones(PA.shape[0]), (PA[:, 0], PA[:, 1])),
    #                      shape=(paper_max, author_max),
    #                      dtype=np.float32)
    # PT_s = sp.coo_matrix((np.ones(PT.shape[0]), (PT[:, 0], PT[:, 1])),
    #                      shape=(paper_max, term_max),
    #                      dtype=np.float32)

    features = feat_emb

    labels_raw = np.genfromtxt("{}{}.txt".format(path, label_file),
                               dtype=np.int32)
    # labels_raw[:, 0] -= 1
    # labels_raw[:, 1] -= 1
    labels = labels_raw
    # labels[labels_raw[:, 0]] = labels_raw[:, 1]

    assert paper_max == labels_raw.shape[0]

    reordered = np.random.permutation(np.arange(paper_max))
    total_labeled = labels_raw.shape[0]

    idx_train = reordered[range(int(total_labeled * train_per))]
    idx_val = reordered[range(int(total_labeled * train_per), int(total_labeled * 0.8))]
    idx_test = reordered[range(int(total_labeled * 0.8), total_labeled)]

    folds = {'train':idx_train,'val':idx_val,'test':idx_test}

    return features, labels, folds

def read_mpindex_cora(path="./data/cora/",train_per=0.1):
    label_file = "paper_label"
    feat_emb_file = 'term_emb.npy'
    PT_file = "PT"

    feat_emb = np.load("{}{}".format(path,feat_emb_file))

    PT = np.genfromtxt("{}{}.txt".format(path, PT_file),
                       dtype=np.int32)
    PT[:, 0] -= 1
    PT[:, 1] -= 1

    paper_max = max(PT[:, 0]) + 1
    term_max = max(PT[:, 1]) + 1

    PT_s = sp.coo_matrix((np.ones(PT.shape[0]), (PT[:, 0], PT[:, 1])),
                         shape=(paper_max, term_max),
                         dtype=np.float32)

    # transformer = TfidfTransformer()
    features = PT_s  # AT
    # features = transformer.fit_transform(features)
    # features = np.array(features.todense())
    # features = np.zeros(paper_max).reshape(-1, 1)

    features = count2feat(features,feat_emb)

    labels_raw = np.genfromtxt("{}{}.txt".format(path, label_file),
                               dtype=np.int32)
    labels_raw[:, 0] -= 1

    no_label_mask = labels_raw[:,1] != 0
    labels_raw = labels_raw[no_label_mask]

    print('labels shape: ', labels_raw.shape)

    #remap label
    label_dict={}
    for i in labels_raw[:,1]:
        if i not in label_dict:
            label_dict[i]=len(label_dict)
    print('number of label classes: ', len(label_dict))
    for i in range(labels_raw.shape[0]):
        labels_raw[i,1] = label_dict[labels_raw[i,1]]

    labels = np.zeros(paper_max)
    labels[labels_raw[:, 0]] = labels_raw[:, 1]

    reordered = np.random.permutation(labels_raw[:, 0])
    total_labeled = labels_raw.shape[0]

    idx_train = reordered[range(int(total_labeled * train_per))]
    idx_val = reordered[range(int(total_labeled * train_per), int(total_labeled * 0.8))]
    idx_test = reordered[range(int(total_labeled * 0.8), total_labeled)]

    folds = {'train':idx_train,'val':idx_val,'test':idx_test}

    return features, labels, folds

def load_edge_emb(path, schemes, n_dim=17, n_author=20000):
    data = np.load("{}edge{}.npz".format(path, n_dim))
    index = {}
    emb = {}
    for scheme in schemes:
        # print('number of authors: {}'.format(n_author))
        ind = sp.coo_matrix((np.arange(1,data[scheme].shape[0]+1),
                             (data[scheme][:, 0], data[scheme][:, 1])),
                            shape=(n_author, n_author),
                            dtype=np.long)

        ind = ind + ind.T.multiply(ind.T>ind)
        ind = sparse_mx_to_torch_sparse_tensor(ind)#.to_dense()

        embedding = np.zeros(n_dim, dtype=np.float32)
        embedding = np.vstack((embedding, data[scheme][:, 2:]))
        emb[scheme] = torch.from_numpy(embedding).float()

        index[scheme] = ind.long()
        print('loading edge embedding for {} complete, num of embeddings: {}'.format(scheme,embedding.shape[0]))

    return index, emb


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def read_mpindex_yelp(path="../../data/yelp/",train_per=0.1):
    label_file = "true_cluster"
    feat_file = "attributes"

    # print("{}{}.txt".format(path, PA_file))
    feat = np.genfromtxt("{}{}.txt".format(path, feat_file),
                       dtype=np.float)

    features = feat[:,:2]
    #features = np.zeros((feat.shape[0],1))
    #features = np.eye(feat.shape[0])

    labels = np.genfromtxt("{}{}.txt".format(path, label_file),
                               dtype=np.int32)

    reordered = np.random.permutation(np.arange(labels.shape[0]))
    total_labeled = labels.shape[0]

    idx_train = reordered[range(int(total_labeled * train_per))]
    idx_val = reordered[range(int(total_labeled * train_per), int(total_labeled * 0.8))]
    idx_test = reordered[range(int(total_labeled * 0.8), total_labeled)]

    folds = {'train':idx_train,'val':idx_val,'test':idx_test}

    return features, labels, folds


def read_mpindex_yago(path="../../data/yago/", label_file = "labels",train_per=0.1):

    movies = []
    with open('{}{}.txt'.format(path, "movies"), mode='r', encoding='UTF-8') as f:
        for line in f:
            movies.append(line.split()[0])

    n_movie = len(movies)
    movie_dict = {a: i for (i, a) in enumerate(movies)}

    features = np.zeros(n_movie).reshape(-1,1)

    labels_raw = []
    with open('{}{}.txt'.format(path, label_file), 'r', encoding='UTF-8') as f:
        for line in f:
            arr = line.split()
            labels_raw.append([int(movie_dict[arr[0]]), int(arr[1])])
    labels_raw = np.asarray(labels_raw)

    labels = np.zeros(n_movie)
    labels[labels_raw[:, 0]] = labels_raw[:, 1]

    reordered = np.random.permutation(labels_raw[:, 0])
    total_labeled = labels_raw.shape[0]

    idx_train = reordered[range(int(total_labeled * train_per))]
    idx_val = reordered[range(int(total_labeled * train_per), int(total_labeled * 0.8))]
    idx_test = reordered[range(int(total_labeled * 0.8), total_labeled)]

    folds = {'train': idx_train, 'val': idx_val, 'test': idx_test}

    return features, labels, folds

def read_homograph(path="../../data/yago/", problem='yago',):
    dataset = "homograph"
    emb_file = {'yago':'MADW_16','dblp':'APC_16','yelp':'RBUK_16'}
    with open("{}{}.emb".format(path, emb_file[problem])) as f:
        n_nodes, n_feature = map(int, f.readline().strip().split())
    embedding = np.loadtxt("{}{}.emb".format(path, emb_file[problem]),
                           dtype=np.float32, skiprows=1, encoding='latin-1')
    emb_index = {}
    for i in range(n_nodes):
        # if type(embedding[i, 0]) is not int:
        #     continue
        emb_index[embedding[i, 0]] = i

    features = np.asarray([embedding[emb_index[i], 1:] for i in range(embedding.shape[0])])
    features = torch.FloatTensor(features)

    edges = np.genfromtxt("{}{}.txt".format(path, dataset),
                          dtype=np.int32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n_nodes, n_nodes),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features

# features, labels, folds = read_mpindex_yago()
