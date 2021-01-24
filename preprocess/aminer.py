import numpy as np
import scipy.sparse as sp
import torch
import random
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm

def gen_homograph():
    path = "data/aminer/"
    out_file = "homograph"

    label_file = "label"
    PA_file = "PA"
    PC_file = "PC"
    # PT_file = "PT"
    PAP_file = "PAP"
    PCP_file = "PCP"
    # APCPA_file = "APCPA"

    PA = np.genfromtxt("{}{}.txt".format(path, PA_file),
                       dtype=np.int32)
    PC = np.genfromtxt("{}{}.txt".format(path, PC_file),
                       dtype=np.int32)

    paper_max = max(PA[:, 0]) + 1
    author_max = max(PA[:, 1]) + 1
    conf_max = max(PC[:, 1]) + 1
    # term_max = max(PT[:, 1]) + 1

    PA[:, 1] += paper_max
    PC[:, 1] += paper_max+author_max
    # PC[:, 1] += author_max+paper_max

    edges = np.concatenate((PA, PC), axis=0)

    np.savetxt("{}{}.txt".format(path, out_file), edges, fmt='%u')


def read_embed(path="../../../data/aminer/",
               emb_file="APC", emb_len=16):
    #========For n2v embeddings======
    # with open("{}{}_{}.emb".format(path, emb_file,emb_len)) as f:
    #    n_nodes, n_feature = map(int, f.readline().strip().split())

    # embedding = np.loadtxt("{}{}_{}.emb".format(path, emb_file,emb_len),
    #                       dtype=np.float32, skiprows=1)

    #========For mp2v embeddings======
    embedding = []
    with open("{}{}_{}.emb".format(path, emb_file, emb_len)) as f:
        n_nodes, n_feature = map(int, f.readline().strip().split())
        n_nodes -= 1
        for line in f:
            arr = line.strip().split()
            if str(arr[0])[0] == '<':
                continue
            embedding.append([int(str(arr[0])[1:])] +
                             list(map(float, arr[1:])))
    embedding = np.asarray(embedding)

    print(embedding.shape)
    print("number of nodes:{}, embedding size:{}".format(n_nodes, n_feature))

    emb_index = {}
    for i in range(n_nodes):
        emb_index[embedding[i, 0]] = i

    features = np.asarray([embedding[emb_index[i], 1:]
                           if i in emb_index else embedding[0, 1:] for i in range(956635)])

    #assert features.shape[1] == n_feature
    #assert features.shape[0] == n_nodes

    return features, n_nodes, n_feature


def dump_edge_emb(path='data/aminer/', emb_len=16, K=5):
    # dump APA
    PAP_file = "PAP"
    PCP_file = "PCP"

    PAP_e, n_nodes, n_emb = read_embed(
        path=path, emb_file='PAP', emb_len=emb_len)
    PCP_e, n_nodes, n_emb = read_embed(
        path=path, emb_file='PCP', emb_len=emb_len)

    PA_file = "PA"
    PC_file = "PC"

    PA = np.genfromtxt("{}{}.txt".format(path, PA_file),
                       dtype=np.int32)
    PC = np.genfromtxt("{}{}.txt".format(path, PC_file),
                       dtype=np.int32)

    paper_max = max(PA[:, 0]) + 1
    author_max = max(PA[:, 1]) + 1
    conf_max = max(PC[:, 1]) + 1

    PAi = {}
    APi = {}
    PCi = {}
    CPi = {}

    for i in tqdm(range(PA.shape[0])):
        p = PA[i, 0]
        a = PA[i, 1]

        if p not in PAi:
            PAi[p] = set()
        if a not in APi:
            APi[a] = set()

        PAi[p].add(a)
        APi[a].add(p)

    for i in tqdm(range(PC.shape[0])):
        p = PC[i, 0]
        c = PC[i, 1]

        if p not in PCi:
            PCi[p] = set()
        if c not in CPi:
            CPi[c] = set()

        PCi[p].add(c)
        CPi[c].add(p)

    PA = sp.coo_matrix((np.ones(PA.shape[0]), (PA[:, 0], PA[:, 1])),
                       shape=(paper_max, author_max),
                       dtype=np.int32)

    PC = sp.coo_matrix((np.ones(PC.shape[0]), (PC[:, 0], PC[:, 1])),
                       shape=(paper_max, conf_max),
                       dtype=np.int32)

    PAP_ps = (sp.load_npz("{}{}".format(path, 'PAP_ps.npz'))).tolil()
    PCP_ps = (sp.load_npz("{}{}".format(path, 'PCP_ps.npz'))).tolil()
    
    schemes = ['PAP','PCP']
    adjs ={'PAP':PAP_ps,'PCP':PCP_ps}
    rel = {'PAP':PAi,'PCP':PCi}
    inv_rel = {'PAP':APi,'PCP':CPi}
    node_emb = {'PAP':PAP_e,'PCP':PCP_e}
    node_neighs = {}
    edge_embs = {}
    node2edge_idxs = {}
    edge_node_adjs = {}

    
    for s in schemes:
    #--------PAP------------
        aa = adjs[s]
        r = rel[s]
        invr = inv_rel[s]
        node_neigh = np.array([max_n(np.array(aa.data[i]), np.array(
                aa.rows[i]), K) for i in tqdm(range(aa.shape[0]))])
        print("node_neigh.shape ", node_neigh.shape)
        
        emb = []
        counter=0
        node_edge_map = [{} for _ in tqdm(range(node_neigh.shape[0]))]
        node_edge_idx = []
        PAPi = {}
        for p in tqdm(range(aa.shape[0])):
            for a in r[p]:
                if a not in invr:
                    continue
                for v in invr[a]:
                    if v not in node_neigh[p]:
                        continue
                    if p not in PAPi:
                        PAPi[p] = {}

                    if v not in PAPi[p]:
                        PAPi[p][v] = []

                    PAPi[p][v].append(a)
            a1 = p
            edge_idx = []
            for a2 in node_neigh[p]: #np.unique
                tmp = [node_emb[s][p] for p in PAPi[a1][a2]]
                tmp = np.sum(tmp, axis=0)/len(PAPi[a1][a2])
                tmp += node_emb[s][a1]+node_emb[s][a2]
                tmp /= 3
                # if a1 <= a2 or ( a1 not in node_neigh[a2] ):
                if a1 in node_edge_map[a2]:
                    edge_idx.append(node_edge_map[a2][a1])
                    node_edge_map[a1][a2] = node_edge_map[a2][a1]
                elif a2 in node_edge_map[a1]:
                    edge_idx.append(node_edge_map[a1][a2])
                else:
                    edge_idx.append(counter)
                    node_edge_map[a1][a2] = counter
                    counter += 1
                    emb.append(tmp)
            node_edge_idx.append(edge_idx)
        emb = np.asarray(emb)
        node_edge_idx = np.asarray(node_edge_idx)
        print("node_edge_idx.shape", node_edge_idx.shape)
        print('emb.shape',emb.shape)

        print("compute edge embeddings {} complete".format(s))
        
        edge_idx=node_edge_idx

        edge_node_adj = [[]for _ in tqdm(range(counter))]
        for i in tqdm(range(edge_idx.shape[0])):
            for j in range(edge_idx.shape[1]):
                try:
                    edge_node_adj[edge_idx[i, j]].append(i)
                except:
                    print(i,j,edge_idx[i, j],len(edge_node_adj))
        edge_node_adj = [np.unique(i) for i in tqdm(edge_node_adj)]
        print(max(map(len, edge_node_adj)))
        edge_node_adj = np.array(
            [xi if len(xi) == 2 else [xi[0], xi[0]] for xi in tqdm(edge_node_adj)])
        
        edge_node_adj = np.array(edge_node_adj)
        print('edge_node_adj.shape ', edge_node_adj.shape)
        node_neighs[s] = node_neigh
        edge_embs[s] = emb
        node2edge_idxs[s] = edge_idx
        edge_node_adjs[s]=edge_node_adj

    #-----gen edge adj-----
    out_dims = emb_len

    np.savez("{}mp_node_neighs_{}_{}.npz".format(path, K, out_dims),
             PAP=node_neighs['PAP'], PCP=node_neighs['PCP'])
    print('dump npz file {}node_neighs.npz complete'.format(path))

    np.savez("{}mp_node2edge_idxs_{}_{}.npz".format(path, K, out_dims),
             PAP=node2edge_idxs['PAP'], PCP=node2edge_idxs['PCP'])
    print('dump npz file {}node2edge_idxs.npz complete'.format(path))

    np.savez("{}mp_edge_embs_{}_{}.npz".format(path, K, out_dims),
             PAP=edge_embs['PAP'], PCP=edge_embs['PCP'])
    print('dump npz file {}edge_embs.npz complete'.format(path))

    np.savez("{}mp_edge_node_adjs_{}_{}.npz".format(path, K, out_dims),
             PAP=edge_node_adjs['PAP'], PCP=edge_node_adjs['PCP'])
    print('dump npz file {}edge_node_adjs.npz complete'.format(path))

    pass


def pathsim(A):
    value = []
    x, y = A.nonzero()
    for i, j in zip(x, y):
        value.append(2 * A[i, j] / (A[i, i] + A[j, j]))
    return sp.coo_matrix((value, (x, y)))


def gen_homoadj():
    path = "data/aminer/"

    PA_file = "PA"
    PC_file = "PC"
    # PT_file = "PT"

    PA = np.genfromtxt("{}{}.txt".format(path, PA_file),
                       dtype=np.int32)
    PC = np.genfromtxt("{}{}.txt".format(path, PC_file),
                       dtype=np.int32)
    # PT = np.genfromtxt("{}{}.txt".format(path, PT_file),
    #                dtype=np.int32)
    # PA[:, 0] -= 1
    # PA[:, 1] -= 1
    # PC[:, 0] -= 1
    # PC[:, 1] -= 1
    # PT[:, 0] -= 1
    # PT[:, 1] -= 1

    paper_max = max(PA[:, 0]) + 1
    author_max = max(PA[:, 1]) + 1
    conf_max = max(PC[:, 1]) + 1
    # term_max = max(PT[:, 1]) + 1

    PA = sp.coo_matrix((np.ones(PA.shape[0]), (PA[:, 0], PA[:, 1])),
                       shape=(paper_max, author_max),
                       dtype=np.float32)
    PC = sp.coo_matrix((np.ones(PC.shape[0]), (PC[:, 0], PC[:, 1])),
                       shape=(paper_max, conf_max),
                       dtype=np.float32)

    PAP = PA*PA.transpose()
    PCP = PC*PC.transpose()

    PAP = pathsim(PAP)
    PCP = pathsim(PCP)

    sp.save_npz("{}{}".format(path, 'PAP_ps.npz'), PAP)
    sp.save_npz("{}{}".format(path, 'PCP_ps.npz'), PCP)

    #APA = np.hstack([APA.nonzero()[0].reshape(-1,1), APA.nonzero()[1].reshape(-1,1)])
    #APAPA = np.hstack([APAPA.nonzero()[0].reshape(-1,1), APAPA.nonzero()[1].reshape(-1,1)])
    #APCPA = np.hstack([APCPA.nonzero()[0].reshape(-1,1), APCPA.nonzero()[1].reshape(-1,1)])

    #np.savetxt("{}{}.txt".format(path, 'APA'),APA,fmt='%u')
    #np.savetxt("{}{}.txt".format(path, 'APAPA'),APA,fmt='%u')
    #np.savetxt("{}{}.txt".format(path, 'APCPA'),APA,fmt='%u')

# l = 100
def worker_PAP(p):
    l = 100
    global PAi
    global APi
    result = "a{}".format(p)
    for _ in range(int(l/2)):
        a = random.sample(PAi[p], 1)[0]
        p = random.sample(APi[a], 1)[0]
        result += " v{} a{}".format(a, p)
    return result

def worker_PCP(p):
    l = 100
    global PCi
    global CPi
    result = "a{}".format(p)
    for _ in range(int(l/2)):
        c = random.sample(PCi[p], 1)[0]
        p = random.sample(CPi[c], 1)[0]
        result += " v{} a{}".format(c, p)
    return result

def init_pool_pc(pc,cp):
        global PCi
        global CPi
        PCi = pc
        CPi = cp
    
def init_pool_pa(pa,ap):
        global PAi
        global APi
        PAi = pa
        APi = ap

def gen_walk(path='data/aminer/'):
    PAP_file = "PAP"
    PCP_file = "PCP"

    PA_file = "PA"
    PC_file = "PC"

    PA = np.genfromtxt("{}{}.txt".format(path, PA_file),
                       dtype=np.int32)
    PC = np.genfromtxt("{}{}.txt".format(path, PC_file),
                       dtype=np.int32)

    paper_max = max(PA[:, 0]) + 1
    author_max = max(PA[:, 1]) + 1
    conf_max = max(PC[:, 1]) + 1

    PA[:, 1] += paper_max
    PC[:, 1] += paper_max+author_max

    PAi = {}
    APi = {}
    PCi = {}
    CPi = {}

    for i in range(PA.shape[0]):
        p = PA[i, 0]
        a = PA[i, 1]

        if p not in PAi:
            PAi[p] = set()
        if a not in APi:
            APi[a] = set()

        PAi[p].add(a)
        APi[a].add(p)

    for i in range(PC.shape[0]):
        p = PC[i, 0]
        c = PC[i, 1]

        if p not in PCi:
            PCi[p] = set()
        if c not in CPi:
            CPi[c] = set()

        PCi[p].add(c)
        CPi[c].add(p)

    # (1) number of walks per node w: 1000; TOO many
    # (2) walk length l: 100;
    # (3) vector dimension d: 128 (LINE: 128 for each order);
    # (4) neighborhood size k: 7; --default is 5
    # (5) size of negative samples: 5
    # mapping of notation: a:author v:paper i:conference
    l = 100
    w = 1000

    import random
    import multiprocessing
    
    

    

    p = multiprocessing.Pool(192, init_pool_pa(PAi,APi))

    with open("{}{}.walk".format(path, PAP_file), mode='w') as f:
        for _ in tqdm(range(w)):
            for result in p.map(worker_PAP, range(paper_max),chunksize=100):
                # (filename, count) tuples from worker
                f.write('%s\n' % result)
            # for p in PAi:
            #     result = "a{}".format(p)
            #     for _ in range(int(l/2)):
            #         a = random.sample(PAi[p], 1)[0]
            #         p = random.sample(APi[a], 1)[0]
            #         result += " v{} a{}".format(a, p)
            #     f.write(result+"\n")
    p.close()
    p.join()
    
    p = multiprocessing.Pool(192, init_pool_pc(PCi,CPi))
    # gen random walk for meta-path CPC
    with open("{}{}.walk".format(path, PCP_file), mode='w') as f:
        for _ in tqdm(range(w)):
            for result in p.map(worker_PCP, range(paper_max),chunksize=100):
                # (filename, count) tuples from worker
                f.write('%s\n' % result)
            # for p in PCi:
            #     result = "a{}".format(p)
            #     for _ in range(int(l/2)):
            #         c = random.sample(PCi[p], 1)[0]
            #         p = random.sample(CPi[c], 1)[0]
            #         result += " v{} a{}".format(c, p)
            #     f.write(result+"\n")
    # gen random walk for meta-path APAPA
    # with open("{}{}.walk".format(path,APAPA_file),mode='w') as f:
    #    for _ in range(w):
    #        for a in APi:
    #            result="a{}".format(a)
    #            for _ in range(int(l/2)):
    #                p = random.sample(APi[a],1)[0]
    #                a = random.sample(PAi[p],1)[0]
    #                result+=" v{} a{}".format(p,a)
    #            f.write(result+"\n")

    pass


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sparse_mx_to_torch_dense_tensor(sparse_mx):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()


def top_n_idx_sparse(matrix, n):
    '''Return index of top n values in each row of a sparse matrix'''
    top_n_idx = []
    for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
        n_row_pick = min(n, ri - le)
        top_n_idx.append(
            matrix.indices[le + np.argpartition(matrix.data[le:ri], -n_row_pick)[-n_row_pick:]])
    return top_n_idx


def max_n(row_data, row_indices, n):
    # indices=max_n(np.array(arr_ll.data[i]),np.array(arr_ll.rows[i]),2)
    i = row_data.argsort()[-n:]
    # i = row_data.argpartition(-n)[-n:]

    if len(i) < n:
        i = np.concatenate([i, i[np.random.choice(len(i), n - len(i))]])

    top_values = row_data[i]
    top_indices = row_indices[i]
    return top_indices


def rand_n(row_data, row_indices, n):
    # indices=max_n(np.array(arr_ll.data[i]),np.array(arr_ll.rows[i]),2)
    # i = row_data
    # i = row_data.argpartition(-n)[-n:]

    # if len(i) < n:
    i = np.random.choice(len(row_data), n)

    # top_values = row_data[i]
    top_indices = row_indices[i]
    return top_indices


def load_edge_emb(path, schemes, n_dim=17, n_author=5000):
    data = np.load("{}edge{}.npz".format(path, n_dim))
    index = {}
    emb = {}
    for scheme in schemes:

        

        # print('number of authors: {}'.format(n_author))
        ind = sp.coo_matrix((np.arange(1, data[scheme].shape[0] + 1),
                             (data[scheme][:, 0], data[scheme][:, 1])),
                            shape=(n_author, n_author),
                            dtype=np.long).tolil()
        # diag = ind.diagonal()
        # ind = ind - diag
        # ind = ind + ind.transpose() + diag

        # ind = torch.LongTensor(ind)

        ind = ind + ind.T.multiply(ind.T > ind)
        # ind = sparse_mx_to_torch_sparse_tensor(ind)  # .to_dense()

        dict_ind = { }

        embedding = np.zeros(n_dim, dtype=np.float32)
        embedding = np.vstack((embedding, data[scheme][:, 2:]))
        emb[scheme] = torch.from_numpy(embedding).float()

        index[scheme] = ind  # .long()
        print('loading edge embedding for {} complete, num of embeddings: {}'.format(
            scheme, embedding.shape[0]))

    return index, emb


def gen_edge_adj(path='data/dblp2', K=5, ps=True, edge_dim=16):
    """

    Args:
        path:
        K:
        ps: use path sim, or use path count for selecting topk.

    Returns:
        node_neigh:
        edge_idx:
        edge_emb:
        edge_neigh:

    """

    PA_file = "PA"
    PC_file = "PC"

    # print("{}{}.txt".format(path, PA_file))
    PA = np.genfromtxt("{}{}.txt".format(path, PA_file),
                       dtype=np.int32)
    PC = np.genfromtxt("{}{}.txt".format(path, PC_file),
                       dtype=np.int32)

    paper_max = max(PA[:, 0]) + 1
    author_max = max(PA[:, 1]) + 1
    conf_max = max(PC[:, 1])+1

    PA = sp.coo_matrix((np.ones(PA.shape[0]), (PA[:, 0], PA[:, 1])),
                       shape=(paper_max, author_max),
                       dtype=np.int32)
    PC = sp.coo_matrix((np.ones(PC.shape[0]), (PC[:, 0], PC[:, 1])),
                       shape=(paper_max, conf_max),
                       dtype=np.int32)

    PAP_ps = (sp.load_npz("{}{}".format(path, 'PAP_ps.npz'))).tolil()
    PCP_ps = (sp.load_npz("{}{}".format(path, 'PCP_ps.npz'))).tolil()
    adj = {'PAP': PAP_ps, 'PCP': PCP_ps}

    # select top-K path-count neighbors of each A. If number of neighbors>K, trunc; else upsampling
    schemes = ['PAP', 'PCP']
    index, emb = load_edge_emb(
        path, schemes, n_dim=edge_dim, n_author=paper_max)

    node_neighs = {}
    edge_neighs = {}
    node2edge_idxs = {}
    edge_embs = {}
    edge2node_idxs = {}
    edge_node_adjs = {}
    for s in schemes:
        print('----{}----'.format(s))
        aa = adj[s]

        # -----Dense matrix version-----
        # count nonzero degree
        # degree = aa.shape[1]-(aa ==0).sum(dim=1)
        # print('min degree ',torch.min(degree))
        # degree = degree.numpy()
        # ind = torch.argsort(aa,dim=1)
        # ind = torch.flip(ind,dims=[1])

        # node_neigh = torch.cat([ind[i, :K].view(1, -1) if degree[i] >= K
        #                         else torch.cat(
        #     [ind[i, :degree[i]], ind[i, np.random.choice(degree[i], K - degree[i])]]).view(1, -1)
        #                         for i in range(ind.shape[0])]
        #                        , dim=0)

        # -----Sparse matrix version-----

        
        node_neigh = np.array([max_n(np.array(aa.data[i]), np.array(
            aa.rows[i]), K) for i in range(aa.shape[0])])
        print("node_neigh.shape ", node_neigh.shape)

        mp_index = (index[s]).to_dense()
        # print(mp_index)
        mp_edge = emb[s]
        out_dims = mp_edge.shape[1]
        edge_idx_old = mp_index[
            torch.arange(node_neigh.shape[0]).repeat_interleave(K).view(-1),
            node_neigh.contiguous().view(-1)]
        print("edge_idx_old.shape ", edge_idx_old.shape)
        old2new = dict()
        new2old = dict()
        for e in edge_idx_old.numpy():
            if e not in old2new:
                old2new[e] = len(old2new)
                new2old[old2new[e]] = e
        assert len(old2new) == len(new2old)
        print('number of unique edges ', len(old2new))
        new_embs = [new2old[i] for i in range(len(old2new))]
        new_embs = mp_edge[new_embs]

        edge_idx = torch.LongTensor([old2new[i]
                                     for i in edge_idx_old.numpy()]).view(-1, K)
        edge_emb = new_embs

        uq = torch.unique(edge_idx.view(-1), return_counts=True)[1]
        print('max number of neighbors ', max(uq))

        # edge->node adj
        edge_node_adj = [[]for _ in range(len(old2new))]
        for i in range(edge_idx.shape[0]):
            for j in range(edge_idx.shape[1]):
                edge_node_adj[edge_idx.numpy()[i, j]].append(i)
        edge_node_adj = [np.unique(i) for i in edge_node_adj]
        edge_node_adj = np.array(
            [xi if len(xi) == 2 else [xi[0], xi[0]] for xi in edge_node_adj])
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
        node_neighs[s] = node_neigh
        node2edge_idxs[s] = edge_idx
        edge_embs[s] = edge_emb
        # edge2node_idxs[s] = edge2node
        edge_node_adjs[s] = edge_node_adj

    # np.savez("{}edge_neighs_{}_{}.npz".format(path,K,out_dims),
    #          APA=edge_neighs['APA'], APAPA=edge_neighs['APAPA'], APCPA=edge_neighs['APCPA'])
    # print('dump npz file {}edge_neighs.npz complete'.format(path))

    np.savez("{}node_neighs_{}_{}.npz".format(path, K, out_dims),
             PAP=node_neighs['PAP'], PCP=node_neighs['PCP'])
    print('dump npz file {}node_neighs.npz complete'.format(path))

    np.savez("{}node2edge_idxs_{}_{}.npz".format(path, K, out_dims),
             PAP=node2edge_idxs['PAP'], PCP=node2edge_idxs['PCP'])
    print('dump npz file {}node2edge_idxs.npz complete'.format(path))

    np.savez("{}edge_embs_{}_{}.npz".format(path, K, out_dims),
             PAP=edge_embs['PAP'], PCP=edge_embs['PCP'])
    print('dump npz file {}edge_embs.npz complete'.format(path))

    # np.savez("{}edge2node_idxs_{}_{}.npz".format(path,K,out_dims),
    #          APA=edge2node_idxs['APA'], APAPA=edge2node_idxs['APAPA'], APCPA=edge2node_idxs['APCPA'])
    # print('dump npz file {}edge2node_idxs.npz complete'.format(path))

    np.savez("{}edge_node_adjs_{}_{}.npz".format(path, K, out_dims),
             PAP=edge_node_adjs['PAP'], PCP=edge_node_adjs['PCP'])
    print('dump npz file {}edge_node_adjs.npz complete'.format(path))

    pass


def dump_edge_emb_han(path='data/aminer/', emb_len=16, K=128):
    # dump APA
    PAP_file = "PAP"
    PCP_file = "PCP"

    PAP_e, n_nodes, n_emb = read_embed(
        path=path, emb_file='APC', emb_len=emb_len)
    PCP_e, n_nodes, n_emb = read_embed(
        path=path, emb_file='APC', emb_len=emb_len)

    PA_file = "PA"
    PC_file = "PC"

    PA = np.genfromtxt("{}{}.txt".format(path, PA_file),
                       dtype=np.int32)
    PC = np.genfromtxt("{}{}.txt".format(path, PC_file),
                       dtype=np.int32)

    paper_max = max(PA[:, 0]) + 1
    author_max = max(PA[:, 1]) + 1
    conf_max = max(PC[:, 1]) + 1

    PAi = {}
    APi = {}
    PCi = {}
    CPi = {}

    for i in tqdm(range(PA.shape[0])):
        p = PA[i, 0]
        a = PA[i, 1]

        if p not in PAi:
            PAi[p] = set()
        if a not in APi:
            APi[a] = set()

        PAi[p].add(a)
        APi[a].add(p)

    for i in tqdm(range(PC.shape[0])):
        p = PC[i, 0]
        c = PC[i, 1]

        if p not in PCi:
            PCi[p] = set()
        if c not in CPi:
            CPi[c] = set()

        PCi[p].add(c)
        CPi[c].add(p)

    PA = sp.coo_matrix((np.ones(PA.shape[0]), (PA[:, 0], PA[:, 1])),
                       shape=(paper_max, author_max),
                       dtype=np.int32)

    PC = sp.coo_matrix((np.ones(PC.shape[0]), (PC[:, 0], PC[:, 1])),
                       shape=(paper_max, conf_max),
                       dtype=np.int32)

    PAP_ps = (sp.load_npz("{}{}".format(path, 'PAP_ps.npz'))).tolil()
    PCP_ps = (sp.load_npz("{}{}".format(path, 'PCP_ps.npz'))).tolil()
    
    node_neighs = {}
    edge_embs = {}
    node2edge_idxs = {}
    edge_node_adjs = {}
    

    # K = max(np.max(np.diff(PAP_ps.tocsr().indptr)),np.max(np.diff(PCP_ps.tocsr().indptr)))
    print('max degree: ', K)
    #--------PAP------------
    aa = PAP_ps
    node_neigh = np.array([rand_n(np.array(aa.data[i]), np.array(
            aa.rows[i]), K) for i in tqdm(range(aa.shape[0]))])
    print("node_neigh.shape ", node_neigh.shape)
    
    # PAP_emb = []
    # PAPi = {}
    # for p in tqdm(range(PAP_ps.shape[0])):
    #     for a in PAi[p]:
    #         if a not in APi:
    #             continue
    #         for v in APi[a]:
    #             if v not in node_neigh[p]:
    #                 continue
    #             if p not in PAPi:
    #                 PAPi[p] = {}

    #             if v not in PAPi[p]:
    #                 PAPi[p][v] = set()

    #             PAPi[p][v].add(a)
    #     a1 = p
    #     for a2 in node_neigh[p]: #np.unique
    #         tmp = [PAP_e[p] for p in PAPi[a1][a2]]
    #         tmp = np.sum(tmp, axis=0)/len(PAPi[a1][a2])
    #         tmp += PAP_e[a1]+PAP_e[a2]
    #         tmp /= 3
    #         # if a1 <= a2 or ( a1 not in node_neigh[a2] ):
    #         PAP_emb.append(np.concatenate(([a1, a2], tmp,)))
    #     del PAPi[p]
    # PAP_emb = np.asarray(PAP_emb)
    
    # print('PAP_emb.shape',PAP_emb.shape)

    # # ind = sp.coo_matrix((np.arange(1, PAP_emb.shape[0] + 1),
    # #                          (PAP_emb[:, 0], PAP_emb[:, 1])),
    # #                         shape=(paper_max, paper_max),
    # #                         dtype=np.long)
    # # ind = ind + ind.T.multiply(ind.T > ind)
    # # ind = sparse_mx_to_torch_sparse_tensor(ind)
    # print("compute edge embeddings {} complete".format(PAP_file))
    # edge_idx_old = torch.arange(node_neigh.shape[0]*node_neigh.shape[1],dtype=torch.long).reshape(-1,K)
    # # edge_idx_old = torch.LongTensor([ ind[i,j
    # #         ] for i,j in tqdm(zip(torch.arange(node_neigh.shape[0]).repeat_interleave(K).view(-1),
    # #         torch.from_numpy(node_neigh).view(-1)))]).reshape(-1,K)

    # print("edge_idx_old.shape ", edge_idx_old.shape)
    # #-------dblp.py-----------
    # old2new = dict()
    # new2old = dict()
    # for e in edge_idx_old.numpy():
    #     if e not in old2new:
    #         old2new[e] = len(old2new)
    #         new2old[old2new[e]] = e
    # assert len(old2new) == len(new2old)
    # print('number of unique edges ', len(old2new))
    # new_embs = [new2old[i] for i in range(len(old2new))]
    # new_embs = mp_edge[new_embs]

    # 
    # edge_idx = torch.LongTensor([old2new[i]
    #                                 for i in tqdm(edge_idx_old.numpy())]).view(-1, K)
    # print("edge_idx.shape ", edge_idx.shape)
    # edge_emb = new_embs

    # uq = torch.unique(edge_idx.view(-1), return_counts=True)[1]
    # print('max number of neighbors ', max(uq))

    # # edge->node adj
    # edge_node_adj = [[]for _ in tqdm(range(len(old2new)))]
    # for i in tqdm(range(edge_idx.shape[0])):
    #     for j in range(edge_idx.shape[1]):
    #         edge_node_adj[edge_idx.numpy()[i, j]].append(i)
    # edge_node_adj = [np.unique(i) for i in tqdm(edge_node_adj)]
    # edge_node_adj = np.array(
    #     [xi if len(xi) == 2 else [xi[0], xi[0]] for xi in tqdm(edge_node_adj)])
    # -------end of dblp.py-----------
    
    # edge_idx=edge_idx_old

    # edge_node_adj = [[]for _ in tqdm(range(node_neigh.shape[0]*node_neigh.shape[1]))]
    # for i in tqdm(range(edge_idx.shape[0])):
    #     for j in range(edge_idx.shape[1]):
    #         try:
    #             edge_node_adj[edge_idx.numpy()[i, j]].append(i)
    #             edge_node_adj[edge_idx.numpy()[i, j]].append(j)
    #         except:
    #             print(i,j,edge_idx.numpy()[i, j],len(edge_node_adj))
    # edge_node_adj = [np.unique(i) for i in tqdm(edge_node_adj)]
    # edge_node_adj = np.array(
    #     [xi if len(xi) == 2 else [xi[0], xi[0]] for xi in tqdm(edge_node_adj)])
    # print(max(map(len, edge_node_adj)))
    # edge_node_adj = np.array(edge_node_adj)
    # print('edge_node_adj.shape ', edge_node_adj.shape)
    node_neighs['PAP'] = node_neigh
    # edge_embs['PAP'] = PAP_emb[:,2:]
    # node2edge_idxs['PAP'] = edge_idx
    # edge_node_adjs['PAP']=edge_node_adj
    
    #--------PCP------------

    aa = PCP_ps
    node_neigh = np.array([rand_n(np.array(aa.data[i]), np.array(
            aa.rows[i]), K) for i in tqdm(range(aa.shape[0]))])
    print("node_neigh.shape ", node_neigh.shape)
    # PCP_emb = []
    # PCPi = {}
    # for p in tqdm(range(PCP_ps.shape[0])):
    #     for a in PCi[p]:
    #         if a not in CPi:
    #             continue
    #         for v in CPi[a]:
    #             if v not in node_neigh[p]:
    #                 continue
    #             if p not in PCPi:
    #                 PCPi[p] = {}

    #             if v not in PCPi[p]:
    #                 PCPi[p][v] = set()

    #             PCPi[p][v].add(a)
    #     a1 = p
    #     for a2 in node_neigh[p]: #np.unique
    #         tmp = [PCP_e[p] for p in PCPi[a1][a2]]
    #         tmp = np.sum(tmp, axis=0)/len(PCPi[a1][a2])
    #         tmp += PCP_e[a1]+PCP_e[a2]
    #         tmp /= 3
    #         # if a1 <= a2 or ( a1 not in node_neigh[a2] ):
    #         PCP_emb.append(np.concatenate(([a1, a2], tmp,)))
    #     del PCPi[p]
    # PCP_emb = np.asarray(PCP_emb)
    # print('PCP_emb.shape',PCP_emb.shape)
    # print("compute edge embeddings {} complete".format(PCP_file))
    # edge_idx_old = torch.arange(node_neigh.shape[0]*node_neigh.shape[1],dtype=torch.long).reshape(-1,K)

    # print("edge_idx_old.shape ", edge_idx_old.shape)
    # edge_idx=edge_idx_old

    # edge_node_adj = [[]for _ in tqdm(range(node_neigh.shape[0]*node_neigh.shape[1]))]
    # for i in tqdm(range(edge_idx.shape[0])):
    #     for j in range(edge_idx.shape[1]):
    #         try:
    #             edge_node_adj[edge_idx.numpy()[i, j]].append(i)
    #             edge_node_adj[edge_idx.numpy()[i, j]].append(j)
    #         except:
    #             print(i,j,edge_idx.numpy()[i, j],len(edge_node_adj))
    # edge_node_adj = [np.unique(i) for i in tqdm(edge_node_adj)]
    # edge_node_adj = np.array(
    #     [xi if len(xi) == 2 else [xi[0], xi[0]] for xi in tqdm(edge_node_adj)])
    # print(max(map(len, edge_node_adj)))
    # edge_node_adj = np.array(edge_node_adj)
    # print('edge_node_adj.shape ', edge_node_adj.shape)


    node_neighs['PCP'] = node_neigh
    # edge_embs['PCP'] = PCP_emb[:,2:]
    # node2edge_idxs['PCP'] = edge_idx
    # edge_node_adjs['PCP']=edge_node_adj

    # emb_len = PAP_emb.shape[1]-2
    # np.savez("{}edge{}.npz".format(path, emb_len),
    #          PAP=PAP_emb, PCP=PCP_emb)
    # print('dump npz file {}edge{}.npz complete'.format(path, emb_len))

    #-----gen edge adj-----
    out_dims = 16

    np.savez("{}node_neighs_{}_{}.npz".format(path, K, out_dims),
             PAP=node_neighs['PAP'], PCP=node_neighs['PCP'])
    print('dump npz file {}node_neighs.npz complete'.format(path))

    # np.savez("{}node2edge_idxs_{}_{}.npz".format(path, K, out_dims),
    #          PAP=node2edge_idxs['PAP'], PCP=node2edge_idxs['PCP'])
    # print('dump npz file {}node2edge_idxs.npz complete'.format(path))

    # np.savez("{}edge_embs_{}_{}.npz".format(path, K, out_dims),
    #          PAP=edge_embs['PAP'], PCP=edge_embs['PCP'])
    # print('dump npz file {}edge_embs.npz complete'.format(path))

    # np.savez("{}edge_node_adjs_{}_{}.npz".format(path, K, out_dims),
    #          PAP=edge_node_adjs['PAP'], PCP=edge_node_adjs['PCP'])
    # print('dump npz file {}edge_node_adjs.npz complete'.format(path))

    pass



if __name__ == '__main__':
    # clean_dblp()
    # gen_homograph()
    dump_edge_emb(path='data/aminer/', emb_len=128, K=3)
    # dump_edge_emb_han(path='data/aminer/', emb_len=16, K=128)
    # gen_homoadj()
    # gen_walk(path='data/aminer/')
    # gen_edge_adj(K=5,path='data/aminer/', edge_dim=16)
    # gen_edge_sim_adj(path='../data/dblp2/', K=10,edge_dim=18,sim='cos')
