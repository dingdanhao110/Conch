import numpy as np
import scipy.sparse as sp

def read_embed(path="./data/dblp/",
               emb_file="RUBK"):
    #with open("{}{}.emb".format(path, emb_file)) as f:
    #    n_nodes, n_feature = map(int, f.readline().strip().split())
    #print("number of nodes:{}, embedding size:{}".format(n_nodes, n_feature))

    #embedding = np.loadtxt("{}{}.emb".format(path, emb_file),
    #                       dtype=np.float32, skiprows=1)
    embedding = []
    with open("{}{}.emb".format(path, emb_file)) as f:
        n_nodes, n_feature = map(int, f.readline().strip().split())
        n_nodes-=1
        for line in f:
            arr = line.strip().split()
            if str(arr[0])[0]=='<': continue
            embedding.append([ int(str(arr[0])[1:]) ]+ list(map(float,arr[1:])))
    embedding = np.asarray(embedding)

    print(embedding.shape)
    print("number of nodes:{}, embedding size:{}".format(n_nodes, n_feature))
    emb_index = {}
    for i in range(n_nodes):
        emb_index[embedding[i, 0]] = i

    features = np.asarray([embedding[emb_index[i], 1:] if i in emb_index else embedding[0, 1:] for i in range(37342)])

    #assert features.shape[1] == n_feature
    #assert features.shape[0] == n_nodes

    return features, n_nodes, n_feature

def gen_homograph(path = "../../../data/yelp/", out_file = "homograph"):

    label_file = "attributes"
    RB_file = "RB"
    RK_file = "RK"
    RU_file = "RU"

    RB = np.genfromtxt("{}{}.txt".format(path, RB_file),
                   dtype=np.int32)
    RK = np.genfromtxt("{}{}.txt".format(path, RK_file),
                   dtype=np.int32)
    RU = np.genfromtxt("{}{}.txt".format(path, RU_file),
                   dtype=np.int32)
    RB[:, 0] -= 1
    RB[:, 1] -= 1
    RK[:, 0] -= 1
    RK[:, 1] -= 1
    RU[:, 0] -= 1
    RU[:, 1] -= 1

    rate_max = max(RB[:, 0]) + 1   #33360
    busi_max = max(RB[:, 1]) + 1   #2614
    key_max = max(RK[:, 1]) + 1    #82
    user_max = max(RU[:, 1]) + 1   #1286

    # busi: [0,busi_max)
    # rate: [busi_max,busi_max+rate_max)
    # key: [busi_max+rate_max,busi_max+rate_max+key_max)
    # user: [busi_max+rate_max+key_max,busi_max+rate_max+key_max+user_max)

    RU[:, 0] += busi_max
    RB[:, 0] += busi_max
    RK[:, 0] += busi_max

    RK[:, 1] += busi_max+rate_max

    RU[:, 1] += busi_max+rate_max+key_max

    edges = np.concatenate((RB,RK,RU),axis=0)

    np.savetxt("{}{}.txt".format(path, out_file),edges,fmt='%u')

def dump_yelp_edge_emb(path='../../../data/yelp/'):
    # dump APA
    label_file = "attributes"
    RB_file = "RB"
    RK_file = "RK"
    RU_file = "RU"

    RB = np.genfromtxt("{}{}.txt".format(path, RB_file),
                       dtype=np.int32)
    RK = np.genfromtxt("{}{}.txt".format(path, RK_file),
                       dtype=np.int32)
    RU = np.genfromtxt("{}{}.txt".format(path, RU_file),
                       dtype=np.int32)
    RB[:, 0] -= 1
    RB[:, 1] -= 1
    RK[:, 0] -= 1
    RK[:, 1] -= 1
    RU[:, 0] -= 1
    RU[:, 1] -= 1

    # BR = np.copy(RB[:, [1, 0]])
    # KR = np.copy(RK[:, [1, 0]])
    # UR = np.copy(RU[:, [1, 0]])
    #
    # BR = BR[BR[:, 0].argsort()]
    # KR = KR[KR[:, 0].argsort()]
    # UR = UR[UR[:, 0].argsort()]

    #--
    #build index for 2hop adjs

    RBi={}
    BRi={}
    RKi={}
    KRi={}
    RUi={}
    URi={}

    for i in range(RB.shape[0]):
        r=RB[i,0]
        b=RB[i,1]

        if r not in RBi:
            RBi[r]=set()
        if b not in BRi:
            BRi[b]=set()

        RBi[r].add(b)
        BRi[b].add(r)

    for i in range(RK.shape[0]):
        r=RK[i,0]
        k=RK[i,1]

        if r not in RKi:
            RKi[r]=set()
        if k not in KRi:
            KRi[k]=set()

        RKi[r].add(k)
        KRi[k].add(r)

    for i in range(RU.shape[0]):
        r=RU[i,0]
        u=RU[i,1]

        if r not in RUi:
            RUi[r]=set()
        if u not in URi:
            URi[u]=set()

        RUi[r].add(u)
        URi[u].add(r)

    BRUi={}
    URBi={}

    BRKi={}
    KRBi={}

    for b in BRi:
        for r in BRi[b]:
            if r not in RUi:
                continue
            for u in RUi[r]:
                if b not in BRUi:
                    BRUi[b] ={}
                if u not in URBi:
                    URBi[u] ={}

                if u not in BRUi[b]:
                    BRUi[b][u]=set()
                if b not in URBi[u]:
                    URBi[u][b]=set()

                BRUi[b][u].add(r)
                URBi[u][b].add(r)

    for b in BRi:
        for r in BRi[b]:
            if r not in RKi:
                continue
            for k in RKi[r]:
                if b not in BRKi:
                    BRKi[b]={}
                if k not in KRBi:
                    KRBi[k]={}
                if k not in BRKi[b]:
                    BRKi[b][k]=set()
                if b not in KRBi[k]:
                    KRBi[k][b]=set()
                BRKi[b][k].add(r)
                KRBi[k][b].add(r)


    rate_max = max(RB[:, 0]) + 1  # 33360
    busi_max = max(RB[:, 1]) + 1  # 2614
    key_max = max(RK[:, 1]) + 1  # 82
    user_max = max(RU[:, 1]) + 1  # 1286

    n_busi = busi_max
    BRURB_e, n_nodes, emb_len = read_embed(path=path,emb_file="BRURB_32")
    BRKRB_e, n_nodes, emb_len = read_embed(path=path,emb_file="BRKRB_32")

    BRURB_ps=sp.load_npz("{}{}".format(path, 'BRURB_ps.npz')).todense()
    BRKRB_ps=sp.load_npz("{}{}".format(path, 'BRKRB_ps.npz')).todense()

    # brurb;
    BRURB_emb = []
    for v in range(n_busi):
        result = {}
        count = {}
        if v not in BRUi.keys():
            # print (v)
            continue
        for u in BRUi[v]:
            np1 = len(BRUi[v][u])
            edge1 = [BRURB_e[p] for p in BRUi[v][u]]
            edge1 = np.sum(np.vstack(edge1), axis=0)  # edge1: the emd between v and a1

            for b in URBi[u].keys():
                np2 = len(URBi[u][b])
                edge2 = [BRURB_e[p] for p in URBi[u][b]]
                edge2 = np.sum(np.vstack(edge2), axis=0)  # edge2: the emd between a1 and a2
                if b not in result:
                    result[b] = BRURB_e[u] * (np2 * np1)
                else:
                    result[b] += BRURB_e[u] * (np2 * np1)
                result[b] += edge1 * np2
                result[b] += edge2 * np1
                if b not in count:
                    count[b]=0
                count[b] += np1*np2

        for b in result:
            if v <= b:
                BRURB_emb.append(np.concatenate(([v, b], (result[b]/count[b]+BRURB_e[v]+BRURB_e[b])/5,[BRURB_ps[v,b]], [count[b]])))
    BRURB_emb = np.asarray(BRURB_emb)
    m = np.max(BRURB_emb[:, -1])
    BRURB_emb[:, -1] /= m
    print("compute edge embeddings {} complete".format('BRURB'))

    #  brkrb
    BRKRB_emb = []

    for v in range(n_busi):
        print(v)
        result = {}
        count = {}
        if v not in BRKi.keys():
            # print (v)
            continue
        for k in BRKi[v].keys():
            np1 = len(BRKi[v][k])
            edge1 = [BRKRB_e[p] for p in BRKi[v][k]]
            edge1 = np.sum(np.vstack(edge1), axis=0)  # edge1: the emd between v and a1

            for b in KRBi[k].keys():
                np2 = len(KRBi[k][b])
                edge2 = [BRKRB_e[p] for p in KRBi[k][b]]
                edge2 = np.sum(np.vstack(edge2), axis=0)  # edge2: the emd between a1 and a2
                if b not in result:
                    result[b] = BRKRB_e[k] * (np2 * np1)
                else:
                    result[b] += BRKRB_e[k] * (np2 * np1)
                if b not in count:
                    count[b]=0
                result[b] += edge1 * np2
                result[b] += edge2 * np1
                count[b] += np1*np2
        for b in result:
            if v <= b:
                BRKRB_emb.append(np.concatenate(([v, b], (result[b]/count[b]+BRKRB_e[v]+BRKRB_e[b])/5,[BRKRB_ps[v,b]], [count[b]] )))
    BRKRB_emb = np.asarray(BRKRB_emb)
    m = np.max(BRKRB_emb[:, -1])
    BRKRB_emb[:, -1] /= m
    print("compute edge embeddings {} complete".format('BRKRB'))
    emb_len = BRKRB_emb.shape[1] - 2
    np.savez("{}edge{}.npz".format(path, emb_len),
             BRURB=BRURB_emb, BRKRB=BRKRB_emb)
    print('dump npz file {}edge{}.npz complete'.format(path, emb_len))
    pass

def pathsim(A):
    value = []
    x,y = A.nonzero()
    for i,j in zip(x,y):
        value.append(2 * A[i, j] / (A[i, i] + A[j, j]))
    return sp.coo_matrix((value,(x,y)))

def gen_homoadj(path = "data/yelp/", out_file = "homograph"):

    label_file = "attributes"
    RB_file = "RB"
    RK_file = "RK"
    RU_file = "RU"

    RB = np.genfromtxt("{}{}.txt".format(path, RB_file),
                   dtype=np.int32)
    RK = np.genfromtxt("{}{}.txt".format(path, RK_file),
                   dtype=np.int32)
    RU = np.genfromtxt("{}{}.txt".format(path, RU_file),
                   dtype=np.int32)
    RB[:, 0] -= 1
    RB[:, 1] -= 1
    RK[:, 0] -= 1
    RK[:, 1] -= 1
    RU[:, 0] -= 1
    RU[:, 1] -= 1

    rate_max = max(RB[:, 0]) + 1   #33360
    busi_max = max(RB[:, 1]) + 1   #2614
    key_max = max(RK[:, 1]) + 1    #82
    user_max = max(RU[:, 1]) + 1   #1286

    # busi: [0,busi_max)
    # rate: [busi_max,busi_max+rate_max)
    # key: [busi_max+rate_max,busi_max+rate_max+key_max)
    # user: [busi_max+rate_max+key_max,busi_max+rate_max+key_max+user_max)

    RB = sp.coo_matrix((np.ones(RB.shape[0]), (RB[:, 0], RB[:, 1])),
                       shape=(rate_max, busi_max),
                       dtype=np.float32)
    RK = sp.coo_matrix((np.ones(RK.shape[0]), (RK[:, 0], RK[:, 1])),
                       shape=(rate_max, key_max),
                       dtype=np.float32)
    RU = sp.coo_matrix((np.ones(RU.shape[0]), (RU[:, 0], RU[:, 1])),
                       shape=(rate_max, user_max),
                       dtype=np.float32)

    BRURB = RB.transpose()*RU*RU.transpose()*RB
    BRKRB = RB.transpose()*RK*RK.transpose()*RB

    BRURB = pathsim(BRURB)
    BRKRB = pathsim(BRKRB)

    sp.save_npz("{}{}".format(path, 'BRURB_ps.npz'), BRURB)
    sp.save_npz("{}{}".format(path, 'BRKRB_ps.npz'), BRKRB)

    #BRURB = np.hstack([BRURB.nonzero()[0].reshape(-1,1), BRURB.nonzero()[1].reshape(-1,1)])
    #BRKRB = np.hstack([BRKRB.nonzero()[0].reshape(-1,1), BRKRB.nonzero()[1].reshape(-1,1)])

    
    #np.savetxt("{}{}.txt".format(path, 'BRURB'),BRURB,fmt='%u')
    #np.savetxt("{}{}.txt".format(path, 'BRKRB'),BRKRB,fmt='%u')

def gen_walk(path='../../../data/yelp/',
                        walk_length=100,n_walks=1000):
    RB_file = "RB"
    RK_file = "RK"
    RU_file = "RU"

    RB = np.genfromtxt("{}{}.txt".format(path, RB_file),
                       dtype=np.int32)
    RK = np.genfromtxt("{}{}.txt".format(path, RK_file),
                       dtype=np.int32)
    RU = np.genfromtxt("{}{}.txt".format(path, RU_file),
                       dtype=np.int32)
    RB[:, 0] -= 1
    RB[:, 1] -= 1
    RK[:, 0] -= 1
    RK[:, 1] -= 1
    RU[:, 0] -= 1
    RU[:, 1] -= 1

    rate_max = max(RB[:, 0]) + 1   #33360
    busi_max = max(RB[:, 1]) + 1   #2614
    key_max = max(RK[:, 1]) + 1    #82
    user_max = max(RU[:, 1]) + 1   #1286

    # busi: [0,busi_max)
    # rate: [busi_max,busi_max+rate_max)
    # key: [busi_max+rate_max,busi_max+rate_max+key_max)
    # user: [busi_max+rate_max+key_max,busi_max+rate_max+key_max+user_max)

    RU[:, 0] += busi_max
    RB[:, 0] += busi_max
    RK[:, 0] += busi_max

    RK[:, 1] += busi_max+rate_max

    RU[:, 1] += busi_max+rate_max+key_max

    #--
    #build index for 2hop adjs

    RBi={}
    BRi={}
    RKi={}
    KRi={}
    RUi={}
    URi={}

    for i in range(RB.shape[0]):
        r=RB[i,0]
        b=RB[i,1]

        if r not in RBi:
            RBi[r]=set()
        if b not in BRi:
            BRi[b]=set()

        RBi[r].add(b)
        BRi[b].add(r)

    for i in range(RK.shape[0]):
        r=RK[i,0]
        k=RK[i,1]

        if r not in RKi:
            RKi[r]=set()
        if k not in KRi:
            KRi[k]=set()

        RKi[r].add(k)
        KRi[k].add(r)

    for i in range(RU.shape[0]):
        r=RU[i,0]
        u=RU[i,1]

        if r not in RUi:
            RUi[r]=set()
        if u not in URi:
            URi[u]=set()

        RUi[r].add(u)
        URi[u].add(r)
    
    index={}
    index['BR'] = BRi
    index['RB'] = RBi
    index['UR'] = URi
    index['RU'] = RUi
    index['KR'] = KRi
    index['RK'] = RKi

    schemes=["BRURB","BRKRB"]

    for scheme in schemes:
        ind1 = index[scheme[0:2]]
        ind2 = index[scheme[1:3]]
        ind3 = index[scheme[2:4]]
        ind4 = index[scheme[3:5]]
        with open('{}{}.walk'.format(path,scheme),'w') as f:

            for v in ind1:

                for n in range(n_walks):
                    out="a{}".format(v)

                    b = v
                    for w in range(int(walk_length/4)):
                        r = np.random.choice(tuple(ind1[b]))
                        out += " v{}".format(r)
                        u = np.random.choice(tuple(ind2[r]))
                        out += " i{}".format(u)
                        r = np.random.choice(tuple(ind3[u]))
                        out += " v{}".format(r)
                        b = np.random.choice(tuple(ind4[r]))
                        out += " a{}".format(b)

                    f.write(out+"\n")
            pass

        pass
import torch
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_edge_emb(path, schemes, n_dim=17, n_author=5000):
    data = np.load("{}edge{}.npz".format(path, n_dim))
    index = {}
    emb = {}
    for scheme in schemes:
        # print('number of authors: {}'.format(n_author))
        ind = sp.coo_matrix((np.arange(1, data[scheme].shape[0] + 1),
                             (data[scheme][:, 0], data[scheme][:, 1])),
                            shape=(n_author, n_author),
                            dtype=np.long)
        # diag = ind.diagonal()
        # ind = ind - diag
        # ind = ind + ind.transpose() + diag

        # ind = torch.LongTensor(ind)

        ind = ind + ind.T.multiply(ind.T > ind)
        ind = sparse_mx_to_torch_sparse_tensor(ind)  # .to_dense()

        embedding = np.zeros(n_dim, dtype=np.float32)
        embedding = np.vstack((embedding, data[scheme][:, 2:]))
        emb[scheme] = torch.from_numpy(embedding).float()

        index[scheme] = ind.long()
        print('loading edge embedding for {} complete, num of embeddings: {}'.format(scheme, embedding.shape[0]))

    return index, emb

def gen_edge_adj(path='data/yelp/', K=80, ps=True,edge_dim=66):
    """

    Args:
        path:
        K:

    Returns:
        node_neigh:
        edge_idx:
        edge_emb:
        edge_neigh:

    """

    RB_file = "RB"
    RK_file = "RK"
    RU_file = "RU"

    RB = np.genfromtxt("{}{}.txt".format(path, RB_file),
                       dtype=np.int32)
    RK = np.genfromtxt("{}{}.txt".format(path, RK_file),
                       dtype=np.int32)
    RU = np.genfromtxt("{}{}.txt".format(path, RU_file),
                       dtype=np.int32)
    RB[:, 0] -= 1
    RB[:, 1] -= 1
    RK[:, 0] -= 1
    RK[:, 1] -= 1
    RU[:, 0] -= 1
    RU[:, 1] -= 1

    rate_max = max(RB[:, 0]) + 1  # 33360
    busi_max = max(RB[:, 1]) + 1  # 2614
    key_max = max(RK[:, 1]) + 1  # 82
    user_max = max(RU[:, 1]) + 1  # 1286

    # busi: [0,busi_max)
    # rate: [busi_max,busi_max+rate_max)
    # key: [busi_max+rate_max,busi_max+rate_max+key_max)
    # user: [busi_max+rate_max+key_max,busi_max+rate_max+key_max+user_max)

    RB = sp.coo_matrix((np.ones(RB.shape[0]), (RB[:, 0], RB[:, 1])),
                       shape=(rate_max, busi_max),
                       dtype=np.float32)
    RK = sp.coo_matrix((np.ones(RK.shape[0]), (RK[:, 0], RK[:, 1])),
                       shape=(rate_max, key_max),
                       dtype=np.float32)
    RU = sp.coo_matrix((np.ones(RU.shape[0]), (RU[:, 0], RU[:, 1])),
                       shape=(rate_max, user_max),
                       dtype=np.float32)
    if ps:
        BRURB_ps = sparse_mx_to_torch_sparse_tensor(sp.load_npz("{}{}".format(path, 'BRURB_ps.npz'))).to_dense()
        BRKRB_ps = sparse_mx_to_torch_sparse_tensor(sp.load_npz("{}{}".format(path, 'BRKRB_ps.npz'))).to_dense()
        adj = {'BRURB': BRURB_ps, 'BRKRB': BRKRB_ps}
    else:
        BRURB = sparse_mx_to_torch_sparse_tensor(RB.transpose() * RU * RU.transpose() * RB).to_dense().long()
        BRKRB = sparse_mx_to_torch_sparse_tensor(RB.transpose() * RK * RK.transpose() * RB).to_dense().long()
        adj = {'BRURB': BRURB, 'BRKRB': BRKRB}
    #select top-K path-count neighbors of each A. If number of neighbors>K, trunc; else upsampling

    schemes = ['BRURB','BRKRB']#
    index, emb=load_edge_emb(path, schemes, n_dim=edge_dim, n_author=rate_max)

    node_neighs={}
    edge_neighs={}
    node2edge_idxs={}
    edge_embs={}
    edge2node_idxs={}
    edge_node_adjs={}
    for s in schemes:
        print('----{}----'.format(s))
        aa = adj[s]

        #count nonzero degree
        degree = aa.shape[1]-(aa ==0).sum(dim=1)
        print('min degree ',torch.min(degree))

        ind = torch.argsort(aa,dim=1)
        ind = torch.flip(ind,dims=[1])
        degree = degree.numpy()
        node_neigh = torch.cat([ind[i, :K].view(1, -1) if degree[i] >= K
                                else torch.cat(
            [ind[i, :degree[i]], ind[i, np.random.choice(degree[i], K - degree[i])]]).view(1, -1)
                                for i in range(ind.shape[0])]
                               , dim=0)
        print("node_neigh.shape ",node_neigh.shape)

        mp_index = (index[s]).to_dense()
        # print(mp_index)
        mp_edge = emb[s][:,:-2]
        out_dims = mp_edge.shape[1]

        edge_idx_old = mp_index[
            torch.arange(node_neigh.shape[0]).repeat_interleave(K).view(-1),
            node_neigh.contiguous().view(-1)]
        print("edge_idx_old.shape ",edge_idx_old.shape)
        old2new=dict()
        new2old=dict()
        for e in edge_idx_old.numpy():
            if e not in old2new:
                old2new[e]=len(old2new)
                new2old[old2new[e]]=e
        assert len(old2new)==len(new2old)
        print('number of unique edges ', len(old2new))
        new_embs = [ new2old[i] for i in range(len(old2new))]
        new_embs = mp_edge[new_embs]

        edge_idx = torch.LongTensor( [old2new[i] for i in edge_idx_old.numpy()]).view(-1,K)
        edge_emb = new_embs

        uq = torch.unique(edge_idx.view(-1),return_counts=True)[1]
        print ('max number of neighbors ', max(uq))

        #edge->node adj
        edge_node_adj = [[]for _ in range(len(old2new))]
        for i in range(edge_idx.shape[0]):
            for j in range(edge_idx.shape[1]):
                edge_node_adj[  edge_idx.numpy()[i,j] ].append(i)
        edge_node_adj = [np.unique(i) for i in edge_node_adj]
        edge_node_adj = np.array([xi if len(xi)==2 else [xi[0],xi[0]] for xi in edge_node_adj])
        # print(max(map(len, edge_node_adj)))
        # edge_node_adj = np.array(edge_node_adj)
        print('edge_node_adj.shape ', edge_node_adj.shape)

        #edges of line graph
        # line_graph_edges = torch.cat( [edge_idx.repeat_interleave(K).reshape(-1,1), edge_idx.repeat(K,1).reshape(-1,1),
        #                                torch.arange(node_neigh.shape[0]).repeat_interleave(K*K).view(-1,1)],dim=1).numpy()
        # assert line_graph_edges.shape[1]==3
        # print("line_graph_edges.shape ", line_graph_edges.shape) # [edge1, edge2, node ]

        # #construct line graph
        # import pandas as pd
        # df = pd.DataFrame(line_graph_edges)
        # edge_neigh = df.groupby(0)[1,2].apply(pd.Series.tolist) #group by edge1; [ [e2,n], .. ]

        # max_len = max([len(i) for i in edge_neigh ])

        # edge_neigh_result=[]
        # edge_idx_result=[]
        # for e,neigh in enumerate(edge_neigh):
        #     neigh = np.asarray(neigh)
        #     idx = np.random.choice(neigh.shape[0], max_len)
        #     edge_neigh_result.append(neigh[idx,0])
        #     edge_idx_result.append(neigh[idx,1])
        # edge_neigh = np.vstack(edge_neigh_result)
        # edge2node = np.vstack(edge_idx_result)

        # edge_neighs[s] = edge_neigh
        # node_neighs[s] = node_neigh
        node2edge_idxs[s] = edge_idx
        edge_embs[s] =edge_emb
        # edge2node_idxs[s] = edge2node
        edge_node_adjs[s] = edge_node_adj

    # np.savez("{}edge_neighs_{}_{}.npz".format(path,K,out_dims),
    #          BRURB=edge_neighs['BRURB'], BRKRB=edge_neighs['BRKRB'])
    # print('dump npz file {}edge_neighs.npz complete'.format(path))

    # np.savez("{}node_neighs_{}_{}.npz".format(path,K,out_dims),
    #          BRURB=node_neighs['BRURB'], BRKRB=node_neighs['BRKRB'])
    # print('dump npz file {}node_neighs.npz complete'.format(path))

    np.savez("{}node2edge_idxs_{}_{}.npz".format(path,K,out_dims),
             BRURB=node2edge_idxs['BRURB'], BRKRB=node2edge_idxs['BRKRB'])
    print('dump npz file {}node2edge_idxs.npz complete'.format(path))

    np.savez("{}edge_embs_{}_{}.npz".format(path,K,out_dims),
             BRURB=edge_embs['BRURB'], BRKRB=edge_embs['BRKRB'])
    print('dump npz file {}edge_embs.npz complete'.format(path))

    # np.savez("{}edge2node_idxs_{}_{}.npz".format(path,K,out_dims),
    #          BRURB=edge2node_idxs['BRURB'], BRKRB=edge2node_idxs['BRKRB'])
    # print('dump npz file {}edge2node_idxs.npz complete'.format(path))

    np.savez("{}edge_node_adjs_{}_{}.npz".format(path, K,out_dims),
             BRURB=edge_node_adjs['BRURB'], BRKRB=edge_node_adjs['BRKRB'])
    print('dump npz file {}edge_node_adjs.npz complete'.format(path))

    pass


def gen_edge_adj_random(path='data/yelp/',  ps=True,edge_dim=66):
    """

    Args:
        path:
        K:

    Returns:
        node_neigh:
        edge_idx:
        edge_emb:
        edge_neigh:

    """

    RB_file = "RB"
    RK_file = "RK"
    RU_file = "RU"

    RB = np.genfromtxt("{}{}.txt".format(path, RB_file),
                       dtype=np.int32)
    RK = np.genfromtxt("{}{}.txt".format(path, RK_file),
                       dtype=np.int32)
    RU = np.genfromtxt("{}{}.txt".format(path, RU_file),
                       dtype=np.int32)
    RB[:, 0] -= 1
    RB[:, 1] -= 1
    RK[:, 0] -= 1
    RK[:, 1] -= 1
    RU[:, 0] -= 1
    RU[:, 1] -= 1

    rate_max = max(RB[:, 0]) + 1  # 33360
    busi_max = max(RB[:, 1]) + 1  # 2614
    key_max = max(RK[:, 1]) + 1  # 82
    user_max = max(RU[:, 1]) + 1  # 1286

    # busi: [0,busi_max)
    # rate: [busi_max,busi_max+rate_max)
    # key: [busi_max+rate_max,busi_max+rate_max+key_max)
    # user: [busi_max+rate_max+key_max,busi_max+rate_max+key_max+user_max)

    RB = sp.coo_matrix((np.ones(RB.shape[0]), (RB[:, 0], RB[:, 1])),
                       shape=(rate_max, busi_max),
                       dtype=np.float32)
    RK = sp.coo_matrix((np.ones(RK.shape[0]), (RK[:, 0], RK[:, 1])),
                       shape=(rate_max, key_max),
                       dtype=np.float32)
    RU = sp.coo_matrix((np.ones(RU.shape[0]), (RU[:, 0], RU[:, 1])),
                       shape=(rate_max, user_max),
                       dtype=np.float32)
    if ps:
        BRURB_ps = sparse_mx_to_torch_sparse_tensor(sp.load_npz("{}{}".format(path, 'BRURB_ps.npz'))).to_dense()
        BRKRB_ps = sparse_mx_to_torch_sparse_tensor(sp.load_npz("{}{}".format(path, 'BRKRB_ps.npz'))).to_dense()
        adj = {'BRURB': BRURB_ps, 'BRKRB': BRKRB_ps}
    else:
        BRURB = sparse_mx_to_torch_sparse_tensor(RB.transpose() * RU * RU.transpose() * RB).to_dense().long()
        BRKRB = sparse_mx_to_torch_sparse_tensor(RB.transpose() * RK * RK.transpose() * RB).to_dense().long()
        adj = {'BRURB': BRURB, 'BRKRB': BRKRB}
    #select top-K path-count neighbors of each A. If number of neighbors>K, trunc; else upsampling

    schemes = ['BRURB','BRKRB']#
    index, emb=load_edge_emb(path, schemes, n_dim=edge_dim, n_author=rate_max)

    node_neighs={}
    edge_neighs={}
    node2edge_idxs={}
    edge_embs={}
    edge2node_idxs={}
    edge_node_adjs={}
    max_degree=0
    for s in schemes:
        aa = adj[s]
        degree = aa.shape[1]-(aa ==0).sum(dim=1)
        max_degree=max(max_degree,torch.max(degree).item())
    print('max degree ',max_degree)
    K=max_degree
    for s in schemes:
        print('----{}----'.format(s))
        aa = adj[s]

        #count nonzero degree
        degree = aa.shape[1]-(aa ==0).sum(dim=1)
        print('min degree ',torch.min(degree))

        ind = torch.argsort(aa,dim=1)
        ind = torch.flip(ind,dims=[1])
        degree = degree.numpy()
        node_neigh = torch.cat([ind[i, :K].view(1, -1) if degree[i] >= K
                                else torch.cat(
            [ind[i, :degree[i]], ind[i, np.random.choice(degree[i], K - degree[i])]]).view(1, -1)
                                for i in range(ind.shape[0])]
                               , dim=0)
        print("node_neigh.shape ",node_neigh.shape)

        mp_index = (index[s]).to_dense()
        # print(mp_index)
        mp_edge = emb[s][:,:-2]
        out_dims = mp_edge.shape[1]

        edge_idx_old = mp_index[
            torch.arange(node_neigh.shape[0]).repeat_interleave(K).view(-1),
            node_neigh.contiguous().view(-1)]
        print("edge_idx_old.shape ",edge_idx_old.shape)
        old2new=dict()
        new2old=dict()
        for e in edge_idx_old.numpy():
            if e not in old2new:
                old2new[e]=len(old2new)
                new2old[old2new[e]]=e
        assert len(old2new)==len(new2old)
        print('number of unique edges ', len(old2new))
        new_embs = [ new2old[i] for i in range(len(old2new))]
        new_embs = mp_edge[new_embs]

        edge_idx = torch.LongTensor( [old2new[i] for i in edge_idx_old.numpy()]).view(-1,K)
        edge_emb = new_embs

        uq = torch.unique(edge_idx.view(-1),return_counts=True)[1]
        print ('max number of neighbors ', max(uq))

        #edge->node adj
        edge_node_adj = [[]for _ in range(len(old2new))]
        for i in range(edge_idx.shape[0]):
            for j in range(edge_idx.shape[1]):
                edge_node_adj[  edge_idx.numpy()[i,j] ].append(i)
        edge_node_adj = [np.unique(i) for i in edge_node_adj]
        edge_node_adj = np.array([xi if len(xi)==2 else [xi[0],xi[0]] for xi in edge_node_adj])
        # print(max(map(len, edge_node_adj)))
        # edge_node_adj = np.array(edge_node_adj)
        print('edge_node_adj.shape ', edge_node_adj.shape)

        # #edges of line graph
        # line_graph_edges = torch.cat( [edge_idx.repeat_interleave(K).reshape(-1,1), edge_idx.repeat(K,1).reshape(-1,1),
        #                                torch.arange(node_neigh.shape[0]).repeat_interleave(K*K).view(-1,1)],dim=1).numpy()
        # assert line_graph_edges.shape[1]==3
        # print("line_graph_edges.shape ", line_graph_edges.shape) # [edge1, edge2, node ]

        # #construct line graph
        # import pandas as pd
        # df = pd.DataFrame(line_graph_edges)
        # edge_neigh = df.groupby(0)[1,2].apply(pd.Series.tolist) #group by edge1; [ [e2,n], .. ]

        # max_len = max([len(i) for i in edge_neigh ])

        # edge_neigh_result=[]
        # edge_idx_result=[]
        # for e,neigh in enumerate(edge_neigh):
        #     neigh = np.asarray(neigh)
        #     idx = np.random.choice(neigh.shape[0], max_len)
        #     edge_neigh_result.append(neigh[idx,0])
        #     edge_idx_result.append(neigh[idx,1])
        # edge_neigh = np.vstack(edge_neigh_result)
        # edge2node = np.vstack(edge_idx_result)

        # edge_neighs[s] = edge_neigh
        # node_neighs[s] = node_neigh
        node2edge_idxs[s] = edge_idx
        edge_embs[s] =edge_emb
        # edge2node_idxs[s] = edge2node
        edge_node_adjs[s] = edge_node_adj

    # np.savez("{}edge_neighs_{}_{}.npz".format(path,K,out_dims),
    #          BRURB=edge_neighs['BRURB'], BRKRB=edge_neighs['BRKRB'])
    # print('dump npz file {}edge_neighs.npz complete'.format(path))

    # np.savez("{}node_neighs_{}_{}.npz".format(path,K,out_dims),
    #          BRURB=node_neighs['BRURB'], BRKRB=node_neighs['BRKRB'])
    # print('dump npz file {}node_neighs.npz complete'.format(path))

    np.savez("{}node2edge_idxs_{}_{}.npz".format(path,K,out_dims),
             BRURB=node2edge_idxs['BRURB'], BRKRB=node2edge_idxs['BRKRB'])
    print('dump npz file {}node2edge_idxs.npz complete'.format(path))

    np.savez("{}edge_embs_{}_{}.npz".format(path,K,out_dims),
             BRURB=edge_embs['BRURB'], BRKRB=edge_embs['BRKRB'])
    print('dump npz file {}edge_embs.npz complete'.format(path))

    # np.savez("{}edge2node_idxs_{}_{}.npz".format(path,K,out_dims),
    #          BRURB=edge2node_idxs['BRURB'], BRKRB=edge2node_idxs['BRKRB'])
    # print('dump npz file {}edge2node_idxs.npz complete'.format(path))

    np.savez("{}edge_node_adjs_{}_{}.npz".format(path, K,out_dims),
             BRURB=edge_node_adjs['BRURB'], BRKRB=edge_node_adjs['BRKRB'])
    print('dump npz file {}edge_node_adjs.npz complete'.format(path))

    pass

def gen_edge_sim_adj(path='data/yelp/', K=80,edge_dim=18,sim='cos'):
    """

    Args:
        path:
        K:

    Returns:
        node_neigh:
        edge_idx:
        edge_emb:
        edge_neigh:

    """
    
    BRURB_e, n_nodes, emb_len = read_embed(path=path,emb_file="BRURB_128")
    BRKRB_e, n_nodes, emb_len = read_embed(path=path,emb_file="BRKRB_128")

    RB_file = "RB"
    RK_file = "RK"
    RU_file = "RU"

    RB = np.genfromtxt("{}{}.txt".format(path, RB_file),
                       dtype=np.int32)
    RK = np.genfromtxt("{}{}.txt".format(path, RK_file),
                       dtype=np.int32)
    RU = np.genfromtxt("{}{}.txt".format(path, RU_file),
                       dtype=np.int32)
    RB[:, 0] -= 1
    RB[:, 1] -= 1
    RK[:, 0] -= 1
    RK[:, 1] -= 1
    RU[:, 0] -= 1
    RU[:, 1] -= 1

    rate_max = max(RB[:, 0]) + 1  # 33360
    busi_max = max(RB[:, 1]) + 1  # 2614
    key_max = max(RK[:, 1]) + 1  # 82
    user_max = max(RU[:, 1]) + 1  # 1286

    # busi: [0,busi_max)
    # rate: [busi_max,busi_max+rate_max)
    # key: [busi_max+rate_max,busi_max+rate_max+key_max)
    # user: [busi_max+rate_max+key_max,busi_max+rate_max+key_max+user_max)

    RB = sp.coo_matrix((np.ones(RB.shape[0]), (RB[:, 0], RB[:, 1])),
                       shape=(rate_max, busi_max),
                       dtype=np.float32)
    RK = sp.coo_matrix((np.ones(RK.shape[0]), (RK[:, 0], RK[:, 1])),
                       shape=(rate_max, key_max),
                       dtype=np.float32)
    RU = sp.coo_matrix((np.ones(RU.shape[0]), (RU[:, 0], RU[:, 1])),
                       shape=(rate_max, user_max),
                       dtype=np.float32)
    if sim=='cos':
        from sklearn.metrics.pairwise import cosine_similarity
        BRURB = torch.from_numpy(cosine_similarity(RB.transpose() * RU * RU.transpose() * RB))
        BRKRB = torch.from_numpy(cosine_similarity(RB.transpose() * RK * RK.transpose() * RB))
        adj = {'BRURB': BRURB, 'BRKRB': BRKRB}
    else:
        BRURB = sparse_mx_to_torch_sparse_tensor(RB.transpose() * RU * RU.transpose() * RB).to_dense().long()
        BRKRB = sparse_mx_to_torch_sparse_tensor(RB.transpose() * RK * RK.transpose() * RB).to_dense().long()
        adj = {'BRURB': BRURB, 'BRKRB': BRKRB}
    #select top-K path-count neighbors of each A. If number of neighbors>K, trunc; else upsampling

    schemes = ['BRURB','BRKRB']#
    index, emb=load_edge_emb(path, schemes, n_dim=edge_dim, n_author=rate_max)
    node_emb={'BRURB':BRURB_e,'BRKRB':BRKRB_e}
    node_neighs={}
    edge_neighs={}
    node2edge_idxs={}
    edge_embs={}
    edge2node_idxs={}
    edge_node_adjs={}
    # max_degree=0
    # for s in schemes:
    #     aa = adj[s]
    #     degree = aa.shape[1]-(aa ==0).sum(dim=1)
    #     max_degree=max(max_degree,torch.max(degree).item())
    # print('max degree ',max_degree)
    # K=max_degree
    for s in schemes:
        print('----{}----'.format(s))
        aa = adj[s]
        ne = node_emb[s]

        #count nonzero degree
        degree = aa.shape[1]-(aa ==0).sum(dim=1)
        print('min degree ',torch.min(degree))
        degree = degree.numpy()
        ind = torch.argsort(aa,dim=1)
        ind = torch.flip(ind,dims=[1])

        node_neigh = torch.cat([ind[i, :K].view(1, -1) if degree[i] >= K
                                else torch.cat(
            [ind[i, :degree[i]], ind[i, np.random.choice(degree[i], K - degree[i])]]).view(1, -1)
                                for i in range(ind.shape[0])]
                               , dim=0)
        print("node_neigh.shape ",node_neigh.shape)

        mp_index = (index[s]).to_dense()

        # print(mp_index)
        mp_edge = (emb[s][:,:-2])
        out_dims = mp_edge.shape[1]

        edge_idx_unfold = torch.cat([
            torch.arange(node_neigh.shape[0]).repeat_interleave(K).view(-1,1),
            node_neigh.contiguous().view(-1,1)],dim=1) #shape(-1,2)

        print("edge_idx_unfold.shape ",edge_idx_unfold.shape)
        # max_edge = mp_index.max()
        edgeHash2emb = dict()
        edge2node=[]
        new_embs = []
        edge_idx_new = []
        n_counter = 0
        # counter = 0
        for e in edge_idx_unfold.numpy():
            n1=e[0]
            n2=e[1]

            edge_hash1 = n1*node_neigh.shape[0]+n2
            edge_hash2 = n2*node_neigh.shape[0]+n1
            
            if edge_hash1 in edgeHash2emb or edge_hash2 in edgeHash2emb:
                edge_idx_new.append(edgeHash2emb[edge_hash1])
            else:
                edgeHash2emb[edge_hash1] = len(new_embs)
                edgeHash2emb[edge_hash2] = len(new_embs)
                
                edge_idx_new.append(len(new_embs))
                edge2node.append([n1,n2])

                edge_id = mp_index[n1][n2]
    
                if edge_id==0:
                    #no edge between
                    new_embs.append((ne[n1]+ne[n2])/2)
                    n_counter += 1
                    # edge_idx_old.append(len(edge_idx_old)+max_edge)
                else:
                    new_embs.append(mp_edge[edge_id])
                assert len(new_embs)==len(edge2node)
            
        print('number of empty edges ', n_counter)

        print('number of edges ', len(new_embs))
        edge_idx = np.array(edge_idx_new).reshape(-1,K)
        print('edge_idx.shape ', edge_idx.shape)

        edge_emb = np.vstack(new_embs)
        print('edge_emb.shape ', edge_emb.shape)

        edge_node_adj = np.array(edge2node)
        print('edge_node_adj.shape ', edge_node_adj.shape)

       
        node2edge_idxs[s] = edge_idx
        edge_embs[s] =edge_emb
        edge_node_adjs[s] = edge_node_adj

    np.savez("{}node2edge_idxs_{}_{}_cos.npz".format(path,K,out_dims),
             BRURB=node2edge_idxs['BRURB'], BRKRB=node2edge_idxs['BRKRB'])
    print('dump npz file {}node2edge_idxs_cos.npz complete'.format(path))

    np.savez("{}edge_embs_{}_{}_cos.npz".format(path,K,out_dims),
             BRURB=edge_embs['BRURB'], BRKRB=edge_embs['BRKRB'])
    print('dump npz file {}edge_embs_cos.npz complete'.format(path))

    np.savez("{}edge_node_adjs_{}_{}_cos.npz".format(path, K,out_dims),
             BRURB=edge_node_adjs['BRURB'], BRKRB=edge_node_adjs['BRKRB'])
    print('dump npz file {}edge_node_adjs_cos.npz complete'.format(path))

    pass


if __name__ == '__main__':
    # gen_homograph()
    #dump_yelp_edge_emb(path='../data/yelp/')
    #gen_homoadj()
    #gen_walk(path='../data/yelp/',
    #                        walk_length=100,n_walks=1000)
    gen_edge_adj(path='data/yelp/', K=5,edge_dim=18)
   # gen_edge_sim_adj(path='data/yelp/', K=10,edge_dim=130,sim='cos')
