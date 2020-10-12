import numpy as np
import scipy.sparse as sp
import torch
import random
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm

def gen_homograph():
    path = "../data/cora/"
    out_file = "homograph2"

    label_file = "paper_label"
    PA_file = "PA"
    PP_file = "PP"
    PT_file = "PT"
    PAP_file = "PAP"
    PPP_file = "PPP"

    PA = np.genfromtxt("{}{}.txt".format(path, PA_file),
                       dtype=np.int32)
    PP = np.genfromtxt("{}{}.txt".format(path, PP_file),
                       dtype=np.int32)
    PT = np.genfromtxt("{}{}.txt".format(path, PT_file),
                       dtype=np.int32)

    PA = PA[:, :2]
    PP = PP[:, :2]
    PT = PT[:, :2]

    PA[:, 0] -= 1
    PA[:, 1] -= 1
    PP[:, 0] -= 1
    PP[:, 1] -= 1
    PT[:, 0] -= 1
    PT[:, 1] -= 1
    print('paper id range:', min(PA[:, 0]), max(PA[:, 0]))
    print('paper id range:', min(PP[:, 0]), min(PP[:, 1]))
    print('author id range:', min(PA[:, 1]), max(PA[:, 1]))
    print('term id range:', min(PT[:, 1]), max(PT[:, 1]))

    paper_max = max(PA[:, 0]) + 1
    author_max = max(PA[:, 1]) + 1
    term_max = max(PT[:, 1]) + 1

    PA[:, 1] += paper_max
    # PC[:, 1] += author_max
    PT[:, 1] += author_max + paper_max
    print(PA.shape)
    edges = np.concatenate((PA, PP, PA[:,::-1]), axis=0)

    np.savetxt("{}{}.txt".format(path, out_file), edges, fmt='%u')


def read_embed(path="../../../data/cora/",
               emb_file="AP", emb_len=16):
    with open("{}{}_{}.emb".format(path, emb_file, emb_len)) as f:
        n_nodes, n_feature = map(int, f.readline().strip().split())
    print("number of nodes:{}, embedding size:{}".format(n_nodes, n_feature))

    embedding = np.loadtxt("{}{}_{}.emb".format(path, emb_file, emb_len),
                           dtype=np.float32, skiprows=1)
    emb_index = {}
    for i in range(n_nodes):
        emb_index[int(embedding[i, 0])] = i

    features = np.asarray([embedding[emb_index[i], 1:] if i in emb_index else embedding[0, 1:] for i in range(n_nodes)])

    # assert features.shape[1] == n_feature
    # assert features.shape[0] == n_nodes

    return features, n_nodes, n_feature


def dump_edge_emb(path='../data/cora/', emb_len=16):
    # dump APA
    PAP_file = "PAP"
    PPP_file = "PPP"
    PCP_file = "PCP"

    PAP_e, n_nodes, n_emb = read_embed(path=path, emb_file='AP', emb_len=emb_len)
    PPP_e, n_nodes, n_emb = read_embed(path=path, emb_file='AP', emb_len=emb_len)
    PCP_e, n_nodes, n_emb = read_embed(path=path, emb_file='AP', emb_len=emb_len)

    PA_file = "PA"
    PP_file = "PP"

    PA = np.genfromtxt("{}{}.txt".format(path, PA_file),
                       dtype=np.int32)
    PC = np.genfromtxt("{}{}.txt".format(path, PP_file),
                       dtype=np.int32)
    PA[:, 0] -= 1
    PA[:, 1] -= 1
    PC[:, 0] -= 1
    PC[:, 1] -= 1

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

    PAPi = {}
    PPPi = {}
    PCPi={}

    for v in PAi:
        for a in PAi[v]:
            if a not in APi:
                continue
            for p in APi[a]:
                if p not in PAPi:
                    PAPi[p] = {}
                if v not in PAPi:
                    PAPi[v] = {}

                if v not in PAPi[p]:
                    PAPi[p][v] = set()
                if p not in PAPi[v]:
                    PAPi[v][p] = set()

                PAPi[p][v].add(a)
                PAPi[v][p].add(a)

    for v in PCi:
        for a in PCi[v]:
            if a not in CPi:
                continue
            for p in CPi[a]:
                if p not in PPPi:
                    PPPi[p] = {}
                if v not in PPPi:
                    PPPi[v] = {}

                if v not in PPPi[p]:
                    PPPi[p][v] = set()
                if p not in PPPi[v]:
                    PPPi[v][p] = set()

                PPPi[p][v].add(a)
                PPPi[v][p].add(a)

    for v in CPi:
        for a in CPi[v]:
            if a not in PCi:
                continue
            for p in PCi[a]:
                if p not in PCPi:
                    PCPi[p] = {}
                if v not in PCPi:
                    PCPi[v] = {}

                if v not in PCPi[p]:
                    PCPi[p][v] = set()
                if p not in PCPi[v]:
                    PCPi[v][p] = set()

                PCPi[p][v].add(a)
                PCPi[v][p].add(a)

    PAP_ps = sp.load_npz("{}{}".format(path, 'PAP_ps.npz')).todense()
    PPP_ps = sp.load_npz("{}{}".format(path, 'PPP_ps.npz')).todense()
    PCP_ps = sp.load_npz("{}{}".format(path, 'PCP_ps.npz')).todense()

    # PAP
    APA = PAPi
    APA_emb = []
    for a1 in tqdm(range(19396)):
        if a1 not in APA or len(APA[a1]) == 0:
            APA_emb.append(np.concatenate(([a1, a1], PAP_e[a1], [1], [1])))
            # print('no neighbor')
            continue
        for a2 in APA[a1]:
            tmp = [PAP_e[p] for p in APA[a1][a2]]
            tmp = np.sum(tmp, axis=0) / len(APA[a1][a2])
            tmp += PAP_e[a1] + PAP_e[a2]
            tmp /= 3
            if a1 <= a2:
                APA_emb.append(np.concatenate(([a1, a2], tmp, [PAP_ps[a1, a2]], [len(APA[a1][a2])])))
    PAP_emb = np.asarray(APA_emb)
    print("compute edge embeddings {} complete".format(PAP_file))

    # PPP
    APA = PPPi
    APA_emb = []
    for a1 in tqdm(range(19396)):
        if a1 not in APA or len(APA[a1]) == 0:
            APA_emb.append(np.concatenate(([a1, a1], PPP_e[a1], [1], [1])))
            # print('no neighbor')
            continue
        for a2 in APA[a1]:
            tmp = [PPP_e[p] for p in APA[a1][a2]]
            tmp = np.sum(tmp, axis=0) / len(APA[a1][a2])
            tmp += PPP_e[a1] + PPP_e[a2]
            tmp /= 3
            if a1 <= a2:
                APA_emb.append(np.concatenate(([a1, a2], tmp, [PPP_ps[a1, a2]], [len(APA[a1][a2])])))
    PPP_emb = np.asarray(APA_emb)
    print("compute edge embeddings {} complete".format(PPP_file))

    # PCP
    APA = PCPi
    APA_emb = []
    for a1 in tqdm(range(19396)):
        if a1 not in APA or len(APA[a1]) == 0:
            APA_emb.append(np.concatenate(([a1, a1], PCP_e[a1], [1], [1])))
            # print('no neighbor')
            continue
        for a2 in APA[a1]:
            tmp = [PCP_e[p] for p in APA[a1][a2]]
            tmp = np.sum(tmp, axis=0) / len(APA[a1][a2])
            tmp += PCP_e[a1] + PCP_e[a2]
            tmp /= 3
            if a1 <= a2:
                APA_emb.append(np.concatenate(([a1, a2], tmp, [PCP_ps[a1, a2]], [len(APA[a1][a2])])))
    PCP_emb = np.asarray(APA_emb)
    print("compute edge embeddings {} complete".format(PCP_file))

    emb_len = PPP_emb.shape[1] - 2
    np.savez("{}edge{}.npz".format(path, emb_len),
             PAP=PAP_emb, PPP=PPP_emb, PCP=PCP_emb)
    print('dump npz file {}edge{}.npz complete'.format(path, emb_len))
    pass

def dump_edge_emb_undirected(path='../data/cora/', emb_len=16):
    # dump APA
    PAP_file = "PAP"
    PPP_file = "PPP"

    PAP_e, n_nodes, n_emb = read_embed(path=path, emb_file='AP', emb_len=emb_len)
    PPP_e, n_nodes, n_emb = read_embed(path=path, emb_file='AP', emb_len=emb_len)
    PP_e, n_nodes, n_emb = read_embed(path=path, emb_file='AP', emb_len=emb_len)

    PA_file = "PA"
    PP_file = "PP"

    PA = np.genfromtxt("{}{}.txt".format(path, PA_file),
                       dtype=np.int32)
    PC = np.genfromtxt("{}{}.txt".format(path, PP_file),
                       dtype=np.int32)
    PA[:, 0] -= 1
    PA[:, 1] -= 1
    PC[:, 0] -= 1
    PC[:, 1] -= 1

    # PC = np.vstack([PC, PC[:,::-1]])

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
        if c not in PCi:
            PCi[c] = set()
        if c not in CPi:
            CPi[c] = set()
        if p not in CPi:
            CPi[p] = set()

        PCi[p].add(c)
        PCi[c].add(p)
        CPi[c].add(p)
        CPi[p].add(c)

    PAPi = {}
    PPPi = {}
    PPi=PCi

    for v in PAi:
        for a in PAi[v]:
            if a not in APi:
                continue
            for p in APi[a]:
                if p not in PAPi:
                    PAPi[p] = {}
                if v not in PAPi:
                    PAPi[v] = {}

                if v not in PAPi[p]:
                    PAPi[p][v] = set()
                if p not in PAPi[v]:
                    PAPi[v][p] = set()

                PAPi[p][v].add(a)
                PAPi[v][p].add(a)

    for v in PCi:
        for a in PCi[v]:
            if a not in CPi:
                continue
            for p in CPi[a]:
                if p not in PPPi:
                    PPPi[p] = {}
                if v not in PPPi:
                    PPPi[v] = {}

                if v not in PPPi[p]:
                    PPPi[p][v] = set()
                if p not in PPPi[v]:
                    PPPi[v][p] = set()

                PPPi[p][v].add(a)
                PPPi[v][p].add(a)




    PAP_ps = sp.load_npz("{}{}".format(path, 'PAP_ps.npz')).todense()
    PPP_ps = sp.load_npz("{}{}".format(path, 'PPP_ps.npz')).todense()
    # PP_ps = sp.load_npz("{}{}".format(path, 'PP_ps.npz')).todense()

    # PAP
    APA = PAPi
    APA_emb = []
    for a1 in tqdm(range(19396)):
        if a1 not in APA or len(APA[a1]) == 0:
            APA_emb.append(np.concatenate(([a1, a1], PAP_e[a1], [1], [1])))
            # print('no neighbor')
            continue
        for a2 in APA[a1]:
            tmp = [PAP_e[p] for p in APA[a1][a2]]
            tmp = np.sum(tmp, axis=0) / len(APA[a1][a2])
            tmp += PAP_e[a1] + PAP_e[a2]
            tmp /= 3
            if a1 <= a2:
                APA_emb.append(np.concatenate(([a1, a2], tmp, [PAP_ps[a1, a2]], [len(APA[a1][a2])])))
    PAP_emb = np.asarray(APA_emb)
    print("compute edge embeddings {} complete".format(PAP_file))

    # PPP
    APA = PPPi
    APA_emb = []
    for a1 in tqdm(range(19396)):
        if a1 not in APA or len(APA[a1]) == 0:
            APA_emb.append(np.concatenate(([a1, a1], PPP_e[a1], [1], [1])))
            # print('no neighbor')
            continue
        for a2 in APA[a1]:
            tmp = [PPP_e[p] for p in APA[a1][a2]]
            tmp = np.sum(tmp, axis=0) / len(APA[a1][a2])
            tmp += PPP_e[a1] + PPP_e[a2]
            tmp /= 3
            if a1 <= a2:
                APA_emb.append(np.concatenate(([a1, a2], tmp, [PPP_ps[a1, a2]], [len(APA[a1][a2])])))
    PPP_emb = np.asarray(APA_emb)
    print("compute edge embeddings {} complete".format(PPP_file))

    # pp embedding
    PP_emb = []
    for p in tqdm(range(19396)):
        if p not in PPi or len(PPi[p]) == 0:
            PP_emb.append(np.concatenate(([p, p], PP_e[p], [1], [1])))
            print('no neighbor')
            continue
        for p2 in PPi[p]:
            if p <= p2:
                PP_emb.append(np.concatenate(([p, p2], (PP_e[p]+PP_e[p2])/2, [1], [len(PPi[p])])))


    PP_emb = np.asarray(PP_emb)
    print(PP_emb.shape)
    print("compute edge embeddings {} complete".format(PP_file))

    emb_len = PPP_emb.shape[1] - 2
    np.savez("{}edge{}.npz".format(path, emb_len),
             PAP=PAP_emb, PPP=PPP_emb, PP=PP_emb)
    print('dump npz file {}edge{}.npz complete'.format(path, emb_len))
    pass

def pathsim(A):
    value = []
    x, y = A.nonzero()
    for i, j in zip(x, y):
        value.append(2 * A[i, j] / (A[i, i] + A[j, j]))
    return sp.coo_matrix((value, (x, y)))


def gen_homoadj(path="../data/cora/"):
    PA_file = "PA"
    PP_file = "PP"

    PA = np.genfromtxt("{}{}.txt".format(path, PA_file),
                       dtype=np.int32)
    PP = np.genfromtxt("{}{}.txt".format(path, PP_file),
                       dtype=np.int32)
    PA[:, 0] -= 1
    PA[:, 1] -= 1
    PP[:, 0] -= 1
    PP[:, 1] -= 1

    paper_max = max(PA[:, 0]) + 1
    author_max = max(PA[:, 1]) + 1

    PA = sp.coo_matrix((np.ones(PA.shape[0]), (PA[:, 0], PA[:, 1])),
                       shape=(paper_max, author_max),
                       dtype=np.float32)
    PP = sp.coo_matrix((np.ones(PP.shape[0]), (PP[:, 0], PP[:, 1])),
                       shape=(paper_max, paper_max),
                       dtype=np.float32)

    PP = PP + PP.transpose()

    PAP = PA * PA.transpose()
    PPP = PP * PP.transpose()


    PAP = pathsim(PAP)
    PPP = pathsim(PPP)
    # PP = pathsim(PP)

    sp.save_npz("{}{}".format(path, 'PAP_ps.npz'), PAP)
    sp.save_npz("{}{}".format(path, 'PPP_ps.npz'), PPP)
    # sp.save_npz("{}{}".format(path, 'PP_ps.npz'), PP)

    # APA = np.hstack([APA.nonzero()[0].reshape(-1,1), APA.nonzero()[1].reshape(-1,1)])
    # APAPA = np.hstack([APAPA.nonzero()[0].reshape(-1,1), APAPA.nonzero()[1].reshape(-1,1)])
    # APCPA = np.hstack([APCPA.nonzero()[0].reshape(-1,1), APCPA.nonzero()[1].reshape(-1,1)])

    # np.savetxt("{}{}.txt".format(path, 'APA'),APA,fmt='%u')
    # np.savetxt("{}{}.txt".format(path, 'APAPA'),APA,fmt='%u')
    # np.savetxt("{}{}.txt".format(path, 'APCPA'),APA,fmt='%u')


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
    PC[:, 1] += author_max + paper_max

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

    APAi = {}
    APCi = {}
    CPAi = {}

    for v in APi:
        for p in APi[v]:
            if p not in PAi:
                continue
            for a in PAi[p]:
                if a not in APAi:
                    APAi[a] = {}
                if v not in APAi:
                    APAi[v] = {}

                if v not in APAi[a]:
                    APAi[a][v] = set()
                if a not in APAi[v]:
                    APAi[v][a] = set()

                APAi[a][v].add(p)
                APAi[v][a].add(p)

    for v in APi:
        for p in APi[v]:
            if p not in PCi:
                continue
            for c in PCi[p]:
                if v not in APCi:
                    APCi[v] = {}
                if c not in CPAi:
                    CPAi[c] = {}

                if c not in APCi[v]:
                    APCi[v][c] = set()
                if v not in CPAi[c]:
                    CPAi[c][v] = set()

                CPAi[c][v].add(p)
                APCi[v][c].add(p)

    # (1) number of walks per node w: 1000; TOO many
    # (2) walk length l: 100;
    # (3) vector dimension d: 128 (LINE: 128 for each order);
    # (4) neighborhood size k: 7; --default is 5
    # (5) size of negative samples: 5
    # mapping of notation: a:author v:paper i:conference
    l = 100
    w = 1000

    import random
    # gen random walk for meta-path APCPA
    with open("{}{}.walk".format(path, APCPA_file), mode='w') as f:
        for _ in range(w):
            for a in APi:
                # print(a)
                result = "a{}".format(a)
                for _ in range(int(l / 4)):
                    p = random.sample(APi[a], 1)[0]
                    c = random.sample(PCi[p], 1)[0]
                    result += " v{} i{}".format(p, c)
                    p = random.sample(CPi[c], 1)[0]
                    while p not in PAi:
                        p = random.sample(CPi[c], 1)[0]
                    a = random.sample(PAi[p], 1)[0]
                    result += " v{} a{}".format(p, a)
                f.write(result + "\n")

    # gen random walk for meta-path APA
    with open("{}{}.walk".format(path, APA_file), mode='w') as f:
        for _ in range(w):
            for a in APi:
                result = "a{}".format(a)
                for _ in range(int(l / 2)):
                    p = random.sample(APi[a], 1)[0]
                    a = random.sample(PAi[p], 1)[0]
                    result += " v{} a{}".format(p, a)
                f.write(result + "\n")
    ##gen random walk for meta-path APAPA
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


def gen_edge_adj(path='../data/cora', K=5):
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

    PA_file = "PA"
    PC_file = "PP"

    # print("{}{}.txt".format(path, PA_file))
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

    PA = sp.coo_matrix((np.ones(PA.shape[0]), (PA[:, 0], PA[:, 1])),
                       shape=(paper_max, author_max),
                       dtype=np.int32)
    PC = sp.coo_matrix((np.ones(PC.shape[0]), (PC[:, 0], PC[:, 1])),
                       shape=(paper_max, paper_max),
                       dtype=np.int32)
    PC = PC + PC.transpose()

    PAP = (PA * PA.transpose())
    PPP = (PC * PC.transpose())
    PAP = PAP + (sp.eye(paper_max) > PAP).astype(np.int32)
    PPP = PPP + (sp.eye(paper_max) > PPP).astype(np.int32)
    PP = PC #+ (sp.eye(paper_max) > PC).astype(np.int32)

    PAP = sparse_mx_to_torch_sparse_tensor(PAP).to_dense().long()
    PPP = sparse_mx_to_torch_sparse_tensor(PPP).to_dense().long()
    PP = sparse_mx_to_torch_sparse_tensor(PP).to_dense().long()

    # select top-K path-count neighbors of each A. If number of neighbors>K, trunc; else upsampling
    adj = {'PAP': PAP, 'PPP': PPP, 'PP': PP}
    schemes = [ 'PAP','PPP', 'PP']  #'PAP',
    index, emb = load_edge_emb(path, schemes, n_dim=130, n_author=paper_max)

    node_neighs = {}
    edge_neighs = {}
    node2edge_idxs = {}
    edge_embs = {}
    edge2node_idxs = {}
    edge_node_adjs = {}
    for s in schemes:
        print('----{}----'.format(s))
        aa = adj[s]

        # count nonzero degree
        degree = aa.shape[1] - (aa == 0).sum(dim=1)
        print(degree[0])
        print('min degree ', torch.min(degree))
        print('max degree ', torch.max(degree))
        print('avg degree ', torch.mean(degree.float()))

        for i in range(degree.shape[0]):
            if degree[i]==0:
                aa[i,i]=1
                degree[i]=1
        degree = degree.numpy()
        ind = torch.argsort(aa, dim=1)
        ind = torch.flip(ind, dims=[1])

        node_neigh = torch.cat([ind[i, :K].view(1, -1) if degree[i] >= K
                                else torch.cat([ind[i, :degree[i]], ind[i, np.random.choice(degree[i], K-degree[i])]]).view(1, -1)
                                for i in range(ind.shape[0])]
                               , dim=0)
        print("node_neigh.shape ", node_neigh.shape)

        mp_index = (index[s]).to_dense()
        # print(mp_index)
        mp_edge = emb[s]

        edge_idx_old = mp_index[
            torch.arange(node_neigh.shape[0]).repeat_interleave(K).view(-1),
            node_neigh.contiguous().view(-1)]
        print('max called edge embedding: ',torch.max(torch.unique(edge_idx_old,return_counts=True)[1]))
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

        edge_idx = torch.LongTensor([old2new[i] for i in edge_idx_old.numpy()]).view(-1, K)
        edge_emb = new_embs

        uq = torch.unique(edge_idx.view(-1), return_counts=True)[1]
        print('max number of neighbors ', max(uq))

        # edge->node adj
        edge_node_adj = [[] for _ in range(len(old2new))]
        for i in range(edge_idx.shape[0]):
            for j in range(edge_idx.shape[1]):
                edge_node_adj[edge_idx.numpy()[i, j]].append(i)
        edge_node_adj = [np.unique(i) for i in edge_node_adj]
        edge_node_adj = np.array([xi if len(xi) == 2 else [xi[0], xi[0]] for xi in edge_node_adj])
        # print(max(map(len, edge_node_adj)))
        # edge_node_adj = np.array(edge_node_adj)
        print('edge_node_adj.shape ', edge_node_adj.shape)
        # print(edge_node_adj[0])
        # edges of line graph
        line_graph_edges = torch.cat(
            [edge_idx.repeat_interleave(K).reshape(-1, 1), edge_idx.repeat(K, 1).reshape(-1, 1),
             torch.arange(node_neigh.shape[0]).repeat_interleave(K * K).view(-1, 1)], dim=1).numpy()
        assert line_graph_edges.shape[1] == 3
        print("line_graph_edges.shape ", line_graph_edges.shape)  # [edge1, edge2, node ]

        # construct line graph
        import pandas as pd
        df = pd.DataFrame(line_graph_edges)
        edge_neigh = df.groupby(0)[1, 2].apply(pd.Series.tolist)  # group by edge1; [ [e2,n], .. ]

        max_len = max([len(i) for i in edge_neigh])
        print('max degree of edge: ', max_len)
        print('edge of max degree: ', np.argmax([len(i) for i in edge_neigh]))

        edge_neigh_result = []
        edge_idx_result = []
        for e, neigh in enumerate(edge_neigh):
            neigh = np.asarray(neigh)
            idx = np.random.choice(neigh.shape[0], max_len)
            edge_neigh_result.append(neigh[idx, 0])
            edge_idx_result.append(neigh[idx, 1])
        edge_neigh = np.vstack(edge_neigh_result)
        edge2node = np.vstack(edge_idx_result)

        print("edge_neigh.shape ", edge_neigh.shape)
        print("edge2node.shape ", edge2node.shape)

        edge_neighs[s] = edge_neigh
        node_neighs[s] = node_neigh
        node2edge_idxs[s] = edge_idx
        edge_embs[s] = edge_emb
        edge2node_idxs[s] = edge2node
        edge_node_adjs[s] = edge_node_adj
    #
    np.savez("{}edge_neighs_{}.npz".format(path, K),
             PAP=edge_neighs['PAP'], PPP=edge_neighs['PPP'], PP=edge_neighs['PP'], )
    print('dump npz file {}edge_neighs.npz complete'.format(path))

    np.savez("{}node_neighs_{}.npz".format(path, K),
             PAP=node_neighs['PAP'], PPP=node_neighs['PPP'],PP=node_neighs['PP'])
    print('dump npz file {}node_neighs.npz complete'.format(path))

    np.savez("{}node2edge_idxs_{}.npz".format(path, K),
             PAP=node2edge_idxs['PAP'], PPP=node2edge_idxs['PPP'],PP=node2edge_idxs['PP'])
    print('dump npz file {}node2edge_idxs.npz complete'.format(path))

    np.savez("{}edge_embs_{}.npz".format(path, K),
             PAP=edge_embs['PAP'], PPP=edge_embs['PPP'], PP=edge_embs['PP'])
    print('dump npz file {}edge_embs.npz complete'.format(path))

    np.savez("{}edge2node_idxs_{}.npz".format(path, K),
             PAP=edge2node_idxs['PAP'], PPP=edge2node_idxs['PPP'], PP=edge2node_idxs['PP'])
    print('dump npz file {}edge2node_idxs.npz complete'.format(path))

    np.savez("{}edge_node_adjs_{}.npz".format(path, K),
             PAP=edge_node_adjs['PAP'], PPP=edge_node_adjs['PPP'], PP=edge_node_adjs['PP'])
    print('dump npz file {}edge_node_adjs.npz complete'.format(path))

    pass


if __name__ == '__main__':
    # gen_homograph()
    # gen_homoadj()
    # dump_edge_emb_undirected(emb_len=128)
    gen_edge_adj(path='../data/cora/', K=5)
