import numpy as np
import scipy.sparse as sp
import torch
import random
from sklearn.feature_extraction.text import TfidfTransformer


def clean_dblp(path='./data/dblp/',new_path='./data/dblp2/'):


    label_file = "author_label"
    PA_file = "PA"
    PC_file = "PC"
    PT_file = "PT"

    PA = np.genfromtxt("{}{}.txt".format(path, PA_file),
                       dtype=np.int32)
    PC = np.genfromtxt("{}{}.txt".format(path, PC_file),
                       dtype=np.int32)
    PT = np.genfromtxt("{}{}.txt".format(path, PT_file),
                       dtype=np.int32)
    
    labels_raw = np.genfromtxt("{}{}.txt".format(path, label_file),
                               dtype=np.int32)
    
    A = {}
    for i,a in enumerate(labels_raw[:,0]):
        A[a]=i+1
    print(len(A))
    PA_new = np.asarray([[PA[i,0],A[PA[i,1]]] for i in range(PA.shape[0]) if PA[i,1] in A])
    PC_new = PC
    PT_new = PT

    labels_new = np.asarray([[A[labels_raw[i,0]],labels_raw[i,1]] for i in range(labels_raw.shape[0]) if labels_raw[i,0] in A])

    np.savetxt("{}{}.txt".format(new_path, PA_file),PA_new,fmt='%i')
    np.savetxt("{}{}.txt".format(new_path, PC_file),PC_new,fmt='%i')
    np.savetxt("{}{}.txt".format(new_path, PT_file),PT_new,fmt='%i')
    np.savetxt("{}{}.txt".format(new_path, label_file),labels_new,fmt='%i')

def gen_homograph():
    path = "data/dblp2/"
    out_file = "homograph"

    label_file = "author_label"
    PA_file = "PA"
    PC_file = "PC"
    PT_file = "PT"
    APA_file = "APA"
    APAPA_file = "APAPA"
    APCPA_file = "APCPA"

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
    conf_max = max(PC[:, 1]) + 1
    term_max = max(PT[:, 1]) + 1

    PA[:, 0] += author_max
    PC[:, 0] += author_max
    PC[:, 1] += author_max+paper_max

    edges = np.concatenate((PA,PC),axis=0)

    np.savetxt("{}{}.txt".format(path, out_file),edges,fmt='%u')

def read_embed(path="../../../data/dblp2/",
               emb_file="APC",emb_len=16):
    #with open("{}{}_{}.emb".format(path, emb_file,emb_len)) as f:
    #    n_nodes, n_feature = map(int, f.readline().strip().split())
    #print("number of nodes:{}, embedding size:{}".format(n_nodes, n_feature))
    
    #embedding = np.loadtxt("{}{}_{}.emb".format(path, emb_file,emb_len),
    #                       dtype=np.float32, skiprows=1)
    embedding = []
    with open("{}{}_{}.emb".format(path, emb_file,emb_len)) as f:
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

    features = np.asarray([embedding[emb_index[i], 1:] if i in emb_index else embedding[0, 1:] for i in range(18405)])

    #assert features.shape[1] == n_feature
    #assert features.shape[0] == n_nodes

    return features, n_nodes, n_feature

def dump_edge_emb(path='../data/dblp2/',emb_len=16):
    # dump APA
    APA_file = "APA"
    APAPA_file = "APAPA"
    APCPA_file = "APCPA"

    APA_e,n_nodes,n_emb =read_embed(path=path,emb_file='APA',emb_len=emb_len)
    APCPA_e,n_nodes,n_emb =read_embed(path=path,emb_file='APCPA',emb_len=emb_len)

    PA_file = "PA"
    PC_file = "PC"

    PA = np.genfromtxt("{}{}.txt".format(path, PA_file),
                       dtype=np.int32)
    PC = np.genfromtxt("{}{}.txt".format(path, PC_file),
                       dtype=np.int32)
    PA[:, 0] -= 1
    PA[:, 1] -= 1
    PC[:, 0] -= 1
    PC[:, 1] -= 1

    PAi={}
    APi={}
    PCi={}
    CPi={}

    for i in range(PA.shape[0]):
        p=PA[i,0]
        a=PA[i,1]

        if p not in PAi:
            PAi[p]=set()
        if a not in APi:
            APi[a]=set()

        PAi[p].add(a)
        APi[a].add(p)

    for i in range(PC.shape[0]):
        p=PC[i,0]
        c=PC[i,1]

        if p not in PCi:
            PCi[p]=set()
        if c not in CPi:
            CPi[c]=set()

        PCi[p].add(c)
        CPi[c].add(p)

    APAi={}
    APCi={}
    CPAi={}

    for v in APi:
        for p in APi[v]:
            if p not in PAi:
                continue
            for a in PAi[p]:
                if a not in APAi:
                    APAi[a] ={}
                if v not in APAi:
                    APAi[v] ={}

                if v not in APAi[a]:
                    APAi[a][v]=set()
                if a not in APAi[v]:
                    APAi[v][a]=set()

                APAi[a][v].add(p)
                APAi[v][a].add(p)
    
    for v in APi:
        for p in APi[v]:
            if p not in PCi:
                continue
            for c in PCi[p]:
                if v not in APCi:
                    APCi[v] ={}
                if c not in CPAi:
                    CPAi[c] ={}

                if c not in APCi[v]:
                    APCi[v][c]=set()
                if v not in CPAi[c]:
                    CPAi[c][v]=set()

                CPAi[c][v].add(p)
                APCi[v][c].add(p)



    ## APAPA; vpa1pa2
    #APAPA_emb = []
    #for v in APAi:
    #    result = {}
    #    count = {}
    #    for a1 in APAi[v]:
    #        np1 = len(APAi[v][a1])
    #        edge1 = [node_emb[p] for p in APAi[v][a1]]
    #        edge1 = np.sum(np.vstack(edge1), axis=0)  # edge1: the emd between v and a1

    #        for a2 in APAi[a1].keys():
    #            np2 = len(APAi[a1][a2])
    #            edge2 = [node_emb[p] for p in APAi[a1][a2]]
    #            edge2 = np.sum(np.vstack(edge2), axis=0)  # edge2: the emd between a1 and a2
    #            if a2 not in result:
    #                result[a2] = node_emb[a2] * (np2 * np1)
    #            else:
    #                result[a2] += node_emb[a2] * (np2 * np1)
    #            result[a2] += edge1 * np2
    #            result[a2] += edge2 * np1
    #            if a2 not in count:
    #                count[a2]=0
    #            count[a2] += np1*np2

    #    for a2 in result:
    #        if v <= a2:
    #            APAPA_emb.append(np.concatenate(([v, a2], result[a2]/count[a2], [count[a2]])))
    #APAPA_emb = np.asarray(APAPA_emb)
    #m = np.max(APAPA_emb[:, -1])
    #APAPA_emb[:, -1] /= m
    #print("compute edge embeddings {} complete".format('APAPA'))    

    APA_ps=sp.load_npz("{}{}".format(path, 'APA_ps.npz')).todense()
    APAPA_ps=sp.load_npz("{}{}".format(path, 'APAPA_ps.npz')).todense()
    APCPA_ps=sp.load_npz("{}{}".format(path, 'APCPA_ps.npz')).todense()

    # APA
    APA = APAi

    APA_emb = []
    for a1 in APA.keys():
        for a2 in APA[a1]:
            tmp = [APA_e[p] for p in APA[a1][a2]]
            tmp = np.sum(tmp, axis=0)/len(APA[a1][a2])
            tmp += APA_e[a1]+APA_e[a2]
            tmp /= 3
            if a1 <= a2:
                APA_emb.append(np.concatenate(([a1, a2], tmp,[APA_ps[a1,a2]], [len(APA[a1][a2])])))
    APA_emb = np.asarray(APA_emb)
    print("compute edge embeddings {} complete".format(APA_file))

    # APAPA
    APAPA_emb = []
    ind1 = APAi
    ind2 = APAi

    for v in ind1:
        result = {}
        count = {}
        for a1 in ind1[v].keys():
            np1 = len(ind1[v][a1])
            edge1 = [APA_e[p] for p in ind1[v][a1]]
            edge1 = np.sum(np.vstack(edge1), axis=0)  # edge1: the emd between v and a1

            for a2 in ind2[a1].keys():
                np2 = len(ind2[a1][a2])
                edge2 = [APA_e[p] for p in ind2[a1][a2]]
                edge2 = np.sum(np.vstack(edge2), axis=0)  # edge2: the emd between a1 and a2
                if a2 not in result:
                    result[a2] = APA_e[a1] * (np2 * np1)
                else:
                    result[a2] += APA_e[a1] * (np2 * np1)
                result[a2] += edge1 * np2
                result[a2] += edge2 * np1
                if a2 not in count:
                    count[a2]=0
                count[a2] += np1*np2

        for a in result:
            if v <= a:
                APAPA_emb.append(np.concatenate(([v, a], (result[a]/count[a]+APA_e[a]+APA_e[v])/5
                                                 ,[APAPA_ps[v,a]],[count[a]])))
            # f.write('{} {} '.format(v, a))
            # f.write(" ".join(map(str, result[a].numpy())))
            # f.write('\n')
    APAPA_emb = np.asarray(APAPA_emb)
    m = np.max(APAPA_emb[:, -1])
    APAPA_emb[:, -1] /= m
    print("compute edge embeddings {} complete".format(APAPA_file))

    #APCPA
    ind1 = APCi
    ind2 = CPAi
    APCPA_emb = []
    for v in ind1:
        result = {}
        count = {}
        if len(ind1[v]) == 0:
            continue
        for a1 in ind1[v].keys():
            np1 = len(ind1[v][a1])
            edge1 = [APCPA_e[p] for p in ind1[v][a1]]
            edge1 = np.sum(np.vstack(edge1), axis=0)  # edge1: the emd between v and a1

            for a2 in ind2[a1].keys():
                np2 = len(ind2[a1][a2])
                edge2 = [APCPA_e[p] for p in ind2[a1][a2]]
                edge2 = np.sum(np.vstack(edge2), axis=0)  # edge2: the emd between a1 and a2
                if a2 not in result:
                    result[a2] = APCPA_e[a1] * (np2 * np1)
                else:
                    result[a2] += APCPA_e[a1] * (np2 * np1)
                if a2 not in count:
                    count[a2]=0
                result[a2] += edge1 * np2
                result[a2] += edge2 * np1
                count[a2] += np1*np2

        
        for a in result:
            if v <= a:
                if APCPA_ps[v,a]==0: print(v,a)
                APCPA_emb.append(np.concatenate(([v, a], (result[a]/count[a]+APCPA_e[a]+APCPA_e[v])/5,
                                                 [APCPA_ps[v,a]],
                                                 [count[a]])))
            # f.write('{} {} '.format(v,a))
            # f.write(" ".join(map(str, result[a].numpy())))
            # f.write('\n')
    APCPA_emb = np.asarray(APCPA_emb)
    m = np.max(APCPA_emb[:, -1])
    APCPA_emb[:, -1] /= m
    print("compute edge embeddings {} complete".format(APCPA_file))
    emb_len=APA_emb.shape[1]-2
    np.savez("{}edge{}.npz".format(path, emb_len),
             APA=APA_emb, APAPA=APAPA_emb, APCPA=APCPA_emb)
    print('dump npz file {}edge{}.npz complete'.format(path, emb_len))
    pass

def pathsim(A):
    value = []
    x,y = A.nonzero()
    for i,j in zip(x,y):
        value.append(2 * A[i, j] / (A[i, i] + A[j, j]))
    return sp.coo_matrix((value,(x,y)))

def gen_homoadj():
    path = "data/dblp2/"

    PA_file = "PA"
    PC_file = "PC"
    PT_file = "PT"

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
    conf_max = max(PC[:, 1]) + 1
    term_max = max(PT[:, 1]) + 1

    PA = sp.coo_matrix((np.ones(PA.shape[0]), (PA[:, 0], PA[:, 1])),
                       shape=(paper_max, author_max),
                       dtype=np.float32)
    PC = sp.coo_matrix((np.ones(PC.shape[0]), (PC[:, 0], PC[:, 1])),
                       shape=(paper_max, conf_max),
                       dtype=np.float32)
    #PT = sp.coo_matrix((np.ones(PT.shape[0]), (PT[:, 0], PT[:, 1])),
    #                   shape=(paper_max, term_max),
    #                   dtype=np.int32)

    APA = PA.transpose()*PA
    APAPA = APA*APA
    APCPA = PA.transpose()*PC * PC.transpose() * PA

    APA = pathsim(APA)
    APAPA = pathsim(APAPA)
    APCPA = pathsim(APCPA)

    sp.save_npz("{}{}".format(path, 'APA_ps.npz'), APA)
    sp.save_npz("{}{}".format(path, 'APAPA_ps.npz'), APAPA)
    sp.save_npz("{}{}".format(path, 'APCPA_ps.npz'), APCPA)

    #APA = np.hstack([APA.nonzero()[0].reshape(-1,1), APA.nonzero()[1].reshape(-1,1)])
    #APAPA = np.hstack([APAPA.nonzero()[0].reshape(-1,1), APAPA.nonzero()[1].reshape(-1,1)])
    #APCPA = np.hstack([APCPA.nonzero()[0].reshape(-1,1), APCPA.nonzero()[1].reshape(-1,1)])

    #np.savetxt("{}{}.txt".format(path, 'APA'),APA,fmt='%u')
    #np.savetxt("{}{}.txt".format(path, 'APAPA'),APA,fmt='%u')
    #np.savetxt("{}{}.txt".format(path, 'APCPA'),APA,fmt='%u')


def gen_walk(path='data/dblp2/'):
    APA_file = "APA"
    APAPA_file = "APAPA"
    APCPA_file = "APCPA"

    PA_file = "PA"
    PC_file = "PC"

    PA = np.genfromtxt("{}{}.txt".format(path, PA_file),
                       dtype=np.int32)
    PC = np.genfromtxt("{}{}.txt".format(path, PC_file),
                       dtype=np.int32)
    PA[:, 0] -= 1
    PA[:, 1] -= 1
    PC[:, 0] -= 1
    PC[:, 1] -= 1

    paper_max = max(PA[:, 0]) + 1
    author_max = max(PA[:, 1]) + 1
    conf_max = max(PC[:, 1]) + 1

    PA[:, 0] += author_max
    PC[:, 0] += author_max
    PC[:, 1] += author_max+paper_max

    PAi={}
    APi={}
    PCi={}
    CPi={}

    for i in range(PA.shape[0]):
        p=PA[i,0]
        a=PA[i,1]

        if p not in PAi:
            PAi[p]=set()
        if a not in APi:
            APi[a]=set()

        PAi[p].add(a)
        APi[a].add(p)

    for i in range(PC.shape[0]):
        p=PC[i,0]
        c=PC[i,1]

        if p not in PCi:
            PCi[p]=set()
        if c not in CPi:
            CPi[c]=set()

        PCi[p].add(c)
        CPi[c].add(p)

    APAi={}
    APCi={}
    CPAi={}

    for v in APi:
        for p in APi[v]:
            if p not in PAi:
                continue
            for a in PAi[p]:
                if a not in APAi:
                    APAi[a] ={}
                if v not in APAi:
                    APAi[v] ={}

                if v not in APAi[a]:
                    APAi[a][v]=set()
                if a not in APAi[v]:
                    APAi[v][a]=set()

                APAi[a][v].add(p)
                APAi[v][a].add(p)
    
    for v in APi:
        for p in APi[v]:
            if p not in PCi:
                continue
            for c in PCi[p]:
                if v not in APCi:
                    APCi[v] ={}
                if c not in CPAi:
                    CPAi[c] ={}

                if c not in APCi[v]:
                    APCi[v][c]=set()
                if v not in CPAi[c]:
                    CPAi[c][v]=set()

                CPAi[c][v].add(p)
                APCi[v][c].add(p)

    #(1) number of walks per node w: 1000; TOO many
    #(2) walk length l: 100;
    #(3) vector dimension d: 128 (LINE: 128 for each order);
    #(4) neighborhood size k: 7; --default is 5
    #(5) size of negative samples: 5
    #mapping of notation: a:author v:paper i:conference
    l = 100
    w = 1000

    import random
    #gen random walk for meta-path APCPA
    with open("{}{}.walk".format(path,APCPA_file),mode='w') as f:
        for _ in range(w):
            for a in APi:
                #print(a)
                result="a{}".format(a)
                for _ in range(int(l/4)):
                    p = random.sample(APi[a],1)[0]
                    c = random.sample(PCi[p],1)[0]
                    result+=" v{} i{}".format(p,c)
                    p = random.sample(CPi[c],1)[0]
                    while p not in PAi:
                        p = random.sample(CPi[c],1)[0]
                    a = random.sample(PAi[p],1)[0]
                    result+=" v{} a{}".format(p,a)
                f.write(result+"\n")

    #gen random walk for meta-path APA
    with open("{}{}.walk".format(path,APA_file),mode='w') as f:
        for _ in range(w):
            for a in APi:
                result="a{}".format(a)
                for _ in range(int(l/2)):
                    p = random.sample(APi[a],1)[0]
                    a = random.sample(PAi[p],1)[0]
                    result+=" v{} a{}".format(p,a)
                f.write(result+"\n")
    ##gen random walk for meta-path APAPA
    #with open("{}{}.walk".format(path,APAPA_file),mode='w') as f:
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

def gen_edge_adj(path='data/dblp2', K=80, ps=True,edge_dim=66):
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
    PT_file = "PT"

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
    conf_max = max(PC[:,1])+1
    term_max = max(PT[:, 1]) + 1

    PA = sp.coo_matrix((np.ones(PA.shape[0]), (PA[:, 0], PA[:, 1])),
                         shape=(paper_max, author_max),
                         dtype=np.int32)
    PT = sp.coo_matrix((np.ones(PT.shape[0]), (PT[:, 0], PT[:, 1])),
                         shape=(paper_max, term_max),
                         dtype=np.int32)
    PC = sp.coo_matrix((np.ones(PC.shape[0]), (PC[:, 0], PC[:, 1])),
                       shape=(paper_max, conf_max),
                       dtype=np.int32)

    if ps:
        APA_ps = sparse_mx_to_torch_sparse_tensor(sp.load_npz("{}{}".format(path, 'APA_ps.npz'))).to_dense()
        APAPA_ps = sparse_mx_to_torch_sparse_tensor(sp.load_npz("{}{}".format(path, 'APAPA_ps.npz'))).to_dense()
        APCPA_ps = sparse_mx_to_torch_sparse_tensor(sp.load_npz("{}{}".format(path, 'APCPA_ps.npz'))).to_dense()
        adj = {'APA':APA_ps,'APAPA':APAPA_ps,'APCPA':APCPA_ps}
    else:
        APA = (PA.transpose() * PA)
        APAPA = sparse_mx_to_torch_sparse_tensor(APA * APA).to_dense().long()
        APA = sparse_mx_to_torch_sparse_tensor(APA).to_dense().long()
        APCPA = sparse_mx_to_torch_sparse_tensor(PA.transpose() * PC * PC.transpose() * PA).to_dense().long()
        adj = {'APA': APA, 'APAPA': APAPA, 'APCPA': APCPA}

    # select top-K path-count neighbors of each A. If number of neighbors>K, trunc; else upsampling
    schemes = ['APA','APAPA','APCPA']#
    index, emb=load_edge_emb(path, schemes, n_dim=edge_dim, n_author=author_max)

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
        node_neighs[s] = node_neigh
        node2edge_idxs[s] = edge_idx
        edge_embs[s] =edge_emb
        # edge2node_idxs[s] = edge2node
        edge_node_adjs[s] = edge_node_adj

    # np.savez("{}edge_neighs_{}_{}.npz".format(path,K,out_dims),
    #          APA=edge_neighs['APA'], APAPA=edge_neighs['APAPA'], APCPA=edge_neighs['APCPA'])
    # print('dump npz file {}edge_neighs.npz complete'.format(path))

    np.savez("{}node_neighs_{}_{}.npz".format(path,K,out_dims),
             APA=node_neighs['APA'], APAPA=node_neighs['APAPA'], APCPA=node_neighs['APCPA'])
    print('dump npz file {}node_neighs.npz complete'.format(path))

    np.savez("{}node2edge_idxs_{}_{}.npz".format(path,K,out_dims),
             APA=node2edge_idxs['APA'], APAPA=node2edge_idxs['APAPA'], APCPA=node2edge_idxs['APCPA'])
    print('dump npz file {}node2edge_idxs.npz complete'.format(path))

    np.savez("{}edge_embs_{}_{}.npz".format(path,K,out_dims),
             APA=edge_embs['APA'], APAPA=edge_embs['APAPA'], APCPA=edge_embs['APCPA'])
    print('dump npz file {}edge_embs.npz complete'.format(path))

    # np.savez("{}edge2node_idxs_{}_{}.npz".format(path,K,out_dims),
    #          APA=edge2node_idxs['APA'], APAPA=edge2node_idxs['APAPA'], APCPA=edge2node_idxs['APCPA'])
    # print('dump npz file {}edge2node_idxs.npz complete'.format(path))

    np.savez("{}edge_node_adjs_{}_{}.npz".format(path, K,out_dims),
             APA=edge_node_adjs['APA'], APAPA=edge_node_adjs['APAPA'], APCPA=edge_node_adjs['APCPA'])
    print('dump npz file {}edge_node_adjs.npz complete'.format(path))

    pass


def gen_edge_adj_random(path='data/dblp2/', ps=True,edge_dim=66):
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
    PT_file = "PT"

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
    conf_max = max(PC[:,1])+1
    term_max = max(PT[:, 1]) + 1

    PA = sp.coo_matrix((np.ones(PA.shape[0]), (PA[:, 0], PA[:, 1])),
                         shape=(paper_max, author_max),
                         dtype=np.int32)
    PT = sp.coo_matrix((np.ones(PT.shape[0]), (PT[:, 0], PT[:, 1])),
                         shape=(paper_max, term_max),
                         dtype=np.int32)
    PC = sp.coo_matrix((np.ones(PC.shape[0]), (PC[:, 0], PC[:, 1])),
                       shape=(paper_max, conf_max),
                       dtype=np.int32)

    if ps:
        APA_ps = sparse_mx_to_torch_sparse_tensor(sp.load_npz("{}{}".format(path, 'APA_ps.npz'))).to_dense()
        APAPA_ps = sparse_mx_to_torch_sparse_tensor(sp.load_npz("{}{}".format(path, 'APAPA_ps.npz'))).to_dense()
        APCPA_ps = sparse_mx_to_torch_sparse_tensor(sp.load_npz("{}{}".format(path, 'APCPA_ps.npz'))).to_dense()
        adj = {'APA':APA_ps,'APAPA':APAPA_ps,'APCPA':APCPA_ps}
    else:
        APA = (PA.transpose() * PA)
        APAPA = sparse_mx_to_torch_sparse_tensor(APA * APA).to_dense().long()
        APA = sparse_mx_to_torch_sparse_tensor(APA).to_dense().long()
        APCPA = sparse_mx_to_torch_sparse_tensor(PA.transpose() * PC * PC.transpose() * PA).to_dense().long()
        adj = {'APA': APA, 'APAPA': APAPA, 'APCPA': APCPA}

    # select top-K path-count neighbors of each A. If number of neighbors>K, trunc; else upsampling
    schemes = ['APA','APAPA','APCPA']#
    index, emb=load_edge_emb(path, schemes, n_dim=edge_dim, n_author=author_max)

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
    #          APA=edge_neighs['APA'], APAPA=edge_neighs['APAPA'], APCPA=edge_neighs['APCPA'])
    # print('dump npz file {}edge_neighs.npz complete'.format(path))

    # np.savez("{}node_neighs_{}_{}.npz".format(path,K,out_dims),
    #          APA=node_neighs['APA'], APAPA=node_neighs['APAPA'], APCPA=node_neighs['APCPA'])
    # print('dump npz file {}node_neighs.npz complete'.format(path))

    np.savez("{}node2edge_idxs_{}_{}.npz".format(path,K,out_dims),
             APA=node2edge_idxs['APA'], APAPA=node2edge_idxs['APAPA'], APCPA=node2edge_idxs['APCPA'])
    print('dump npz file {}node2edge_idxs.npz complete'.format(path))

    np.savez("{}edge_embs_{}_{}.npz".format(path,K,out_dims),
             APA=edge_embs['APA'], APAPA=edge_embs['APAPA'], APCPA=edge_embs['APCPA'])
    print('dump npz file {}edge_embs.npz complete'.format(path))

    # np.savez("{}edge2node_idxs_{}_{}.npz".format(path,K,out_dims),
    #          APA=edge2node_idxs['APA'], APAPA=edge2node_idxs['APAPA'], APCPA=edge2node_idxs['APCPA'])
    # print('dump npz file {}edge2node_idxs.npz complete'.format(path))

    np.savez("{}edge_node_adjs_{}_{}.npz".format(path, K,out_dims),
             APA=edge_node_adjs['APA'], APAPA=edge_node_adjs['APAPA'], APCPA=edge_node_adjs['APCPA'])
    print('dump npz file {}edge_node_adjs.npz complete'.format(path))

    pass

def gen_edge_sim_adj(path='data/dblp2', K=80,edge_dim=66,sim='cos'):
    """

    Args:
        path:
        K:
        sim: similarity measure. cos:cosine.

    Returns:
        node_neigh:
        edge_idx:
        edge_emb:
        edge_neigh:

    """
    
    APA_e,n_nodes,n_emb =read_embed(path=path,emb_file='APA',emb_len=edge_dim-2)
    APCPA_e,n_nodes,n_emb =read_embed(path=path,emb_file='APCPA',emb_len=edge_dim-2)

    PA_file = "PA"
    PC_file = "PC"
    PT_file = "PT"

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
    conf_max = max(PC[:,1])+1
    term_max = max(PT[:, 1]) + 1

    PA = sp.coo_matrix((np.ones(PA.shape[0]), (PA[:, 0], PA[:, 1])),
                         shape=(paper_max, author_max),
                         dtype=np.int32)
    PT = sp.coo_matrix((np.ones(PT.shape[0]), (PT[:, 0], PT[:, 1])),
                         shape=(paper_max, term_max),
                         dtype=np.int32)
    PC = sp.coo_matrix((np.ones(PC.shape[0]), (PC[:, 0], PC[:, 1])),
                       shape=(paper_max, conf_max),
                       dtype=np.int32)

    if sim=='cos':
        from sklearn.metrics.pairwise import cosine_similarity
        # s = cosine_similarity(self.emd, self.emd) # dense output

        APA = (PA.transpose() * PA)
        APAPA = torch.from_numpy(cosine_similarity(APA * APA))
        APA = torch.from_numpy(cosine_similarity(APA))
        APCPA = torch.from_numpy(cosine_similarity(PA.transpose() * PC * PC.transpose() * PA))

        adj = {'APA': APA, 'APAPA': APAPA, 'APCPA': APCPA}

    else:
        APA = (PA.transpose() * PA)
        APAPA = sparse_mx_to_torch_sparse_tensor(APA * APA).to_dense().long()
        APA = sparse_mx_to_torch_sparse_tensor(APA).to_dense().long()
        APCPA = sparse_mx_to_torch_sparse_tensor(PA.transpose() * PC * PC.transpose() * PA).to_dense().long()
        adj = {'APA': APA, 'APAPA': APAPA, 'APCPA': APCPA}

    # select top-K path-count neighbors of each A. If number of neighbors>K, trunc; else upsampling
    schemes = ['APA','APAPA','APCPA']#
    index, emb=load_edge_emb(path, schemes, n_dim=edge_dim, n_author=author_max)
    node_emb={'APA':APA_e,'APAPA':APA_e,'APCPA':APCPA_e,}

    node_neighs={}
    edge_neighs={}
    node2edge_idxs={}
    edge_embs={}
    edge2node_idxs={}
    edge_node_adjs={}
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
             APA=node2edge_idxs['APA'], APAPA=node2edge_idxs['APAPA'], APCPA=node2edge_idxs['APCPA'])
    print('dump npz file {}node2edge_idxs2.npz complete'.format(path))

    np.savez("{}edge_embs_{}_{}_cos.npz".format(path,K,out_dims),
             APA=edge_embs['APA'], APAPA=edge_embs['APAPA'], APCPA=edge_embs['APCPA'])
    print('dump npz file {}edge_embs2.npz complete'.format(path))

    np.savez("{}edge_node_adjs_{}_{}_cos.npz".format(path, K,out_dims),
             APA=edge_node_adjs['APA'], APAPA=edge_node_adjs['APAPA'], APCPA=edge_node_adjs['APCPA'])
    print('dump npz file {}edge_node_adjs2.npz complete'.format(path))

    pass


if __name__ == '__main__':
    #clean_dblp()
    #gen_homograph()
    # dump_edge_emb(emb_len=128)
    #gen_homoadj()
    #gen_walk(path='../data/dblp2/')
    gen_edge_adj(K=5,path='../data/dblp2/', edge_dim=130)
    # gen_edge_sim_adj(path='../data/dblp2/', K=10,edge_dim=18,sim='cos')
