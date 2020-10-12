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

    features = np.asarray([embedding[emb_index[i], 1:] if i in emb_index else embedding[0, 1:] for i in range(43854)])

    #assert features.shape[1] == n_feature
    #assert features.shape[0] == n_nodes

    return features, n_nodes, n_feature


def gen_homograph(path="../../../data/yago/", out_file="homograph"):
    label_file = "labels"
    MA_file = "movie_actor"
    MD_file = "movie_director"
    MW_file = "movie_writer"

    movies = []
    actors = []
    directors = []
    writers = []

    with open('{}{}.txt'.format(path, "movies"), mode='r', encoding='UTF-8') as f:
        for line in f:
            movies.append(line.split()[0])

    with open('{}{}.txt'.format(path, "actors"), mode='r', encoding='UTF-8') as f:
        for line in f:
            actors.append(line.split()[0])

    with open('{}{}.txt'.format(path, "directors"), mode='r', encoding='UTF-8') as f:
        for line in f:
            directors.append(line.split()[0])

    with open('{}{}.txt'.format(path, "writers"), mode='r', encoding='UTF-8') as f:
        for line in f:
            writers.append(line.split()[0])

    n_movie = len(movies)     #1465
    n_actor = len(actors)    #4019
    n_director = len(directors) #1093
    n_writer = len(writers)    #1458

    movie_dict = {a: i for (i, a) in enumerate(movies)}
    actor_dict = {a: i+n_movie for (i, a) in enumerate(actors)}
    director_dict = {a: i+n_movie+n_actor for (i, a) in enumerate(directors)}
    writer_dict = {a: i+n_movie+n_actor+n_director for (i, a) in enumerate(writers)}


    MA = []
    with open('{}{}.txt'.format(path, MA_file), 'r', encoding='UTF-8') as f:
        for line in f:
            arr = line.split()
            MA.append([movie_dict[arr[0]], actor_dict[arr[1]] ])

    MD = []
    with open('{}{}.txt'.format(path, MD_file), 'r', encoding='UTF-8') as f:
        for line in f:
            arr = line.split()
            MD.append([movie_dict[arr[0]], director_dict[arr[1]]])

    MW = []
    with open('{}{}.txt'.format(path, MW_file), 'r', encoding='UTF-8') as f:
        for line in f:
            arr = line.split()
            MW.append([movie_dict[arr[0]], writer_dict[arr[1]]])

    MA = np.asarray(MA)
    MD = np.asarray(MD)
    MW = np.asarray(MW)

    edges = np.concatenate((MA, MD, MW), axis=0)

    np.savetxt("{}{}.txt".format(path, out_file), edges, fmt='%u')


def dump_yago_edge_emb(path='../../../data/yago/',edge_len=128):
    # dump APA
    label_file = "labels"
    MA_file = "movie_actor"
    MD_file = "movie_director"
    MW_file = "movie_writer"

    movies = []
    actors = []
    directors = []
    writers = []

    with open('{}{}.txt'.format(path, "movies"), mode='r', encoding='UTF-8') as f:
        for line in f:
            movies.append(line.split()[0])

    with open('{}{}.txt'.format(path, "actors"), mode='r', encoding='UTF-8') as f:
        for line in f:
            actors.append(line.split()[0])

    with open('{}{}.txt'.format(path, "directors"), mode='r', encoding='UTF-8') as f:
        for line in f:
            directors.append(line.split()[0])

    with open('{}{}.txt'.format(path, "writers"), mode='r', encoding='UTF-8') as f:
        for line in f:
            writers.append(line.split()[0])

    n_movie = len(movies)  # 3492
    n_actor = len(actors)  # 33401
    n_director = len(directors)  # 2502
    n_writer = len(writers)  # 4459

    print(n_movie,n_actor,n_director,n_writer)

    movie_dict = {a: i for (i, a) in enumerate(movies)}
    actor_dict = {a: i + n_movie for (i, a) in enumerate(actors)}
    director_dict = {a: i + n_movie + n_actor for (i, a) in enumerate(directors)}
    writer_dict = {a: i + n_movie + n_actor + n_director for (i, a) in enumerate(writers)}

    MA = []
    with open('{}{}.txt'.format(path, MA_file), 'r', encoding='UTF-8') as f:
        for line in f:
            arr = line.split()
            MA.append([movie_dict[arr[0]], actor_dict[arr[1]]])

    MD = []
    with open('{}{}.txt'.format(path, MD_file), 'r', encoding='UTF-8') as f:
        for line in f:
            arr = line.split()
            MD.append([movie_dict[arr[0]], director_dict[arr[1]]])

    MW = []
    with open('{}{}.txt'.format(path, MW_file), 'r', encoding='UTF-8') as f:
        for line in f:
            arr = line.split()
            MW.append([movie_dict[arr[0]], writer_dict[arr[1]]])

    MA = np.asarray(MA)
    MD = np.asarray(MD)
    MW = np.asarray(MW)

    #--
    #build index for 2hop adjs

    MAi={}
    MDi={}
    MWi={}
    AMi={}
    DMi={}
    WMi={}

    for i in range(MA.shape[0]):
        m=MA[i,0]
        a=MA[i,1]

        if m not in MAi:
            MAi[m]=set()
        if a not in AMi:
            AMi[a]=set()

        MAi[m].add(a)
        AMi[a].add(m)

    for i in range(MD.shape[0]):
        m = MD[i, 0]
        d = MD[i, 1]

        if m not in MDi:
            MDi[m] = set()
        if d not in DMi:
            DMi[d] = set()

        MDi[m].add(d)
        DMi[d].add(m)

    for i in range(MW.shape[0]):
        m = MW[i, 0]
        w = MW[i, 1]

        if m not in MWi:
            MWi[m] = set()
        if w not in WMi:
            WMi[w] = set()

        MWi[m].add(w)
        WMi[w].add(m)

    MAMi={}
    MDMi={}
    MWMi={}

    for v in MAi:
        for a in MAi[v]:
            if a not in AMi:
                continue
            for m in AMi[a]:
                if m not in MAMi:
                    MAMi[m] ={}
                if v not in MAMi:
                    MAMi[v] ={}

                if v not in MAMi[m]:
                    MAMi[m][v]=set()
                if m not in MAMi[v]:
                    MAMi[v][m]=set()

                MAMi[m][v].add(a)
                MAMi[v][m].add(a)

    for v in MDi:
        for d in MDi[v]:
            if d not in DMi:
                continue
            for m in DMi[d]:
                if m not in MDMi:
                    MDMi[m] = {}
                if v not in MDMi:
                    MDMi[v] = {}

                if v not in MDMi[m]:
                    MDMi[m][v] = set()
                if m not in MDMi[v]:
                    MDMi[v][m] = set()

                MDMi[m][v].add(d)
                MDMi[v][m].add(d)

    for v in MWi:
        for w in MWi[v]:
            if w not in WMi:
                continue
            for m in WMi[w]:
                if m not in MWMi:
                    MWMi[m] ={}
                if v not in MWMi:
                    MWMi[v] ={}

                if v not in MWMi[m]:
                    MWMi[m][v]=set()
                if m not in MWMi[v]:
                    MWMi[v][m]=set()

                MWMi[m][v].add(w)
                MWMi[v][m].add(w)


    MAM_e, n_nodes, emb_len = read_embed(path=path,emb_file="MAM_{}".format(edge_len))
    MDM_e, n_nodes, emb_len = read_embed(path=path,emb_file="MDM_{}".format(edge_len))
    MWM_e, n_nodes, emb_len = read_embed(path=path,emb_file="MWM_{}".format(edge_len))
    #print(n_nodes, emb_len)

    MAM_ps=sp.load_npz("{}{}".format(path, 'MAM_ps.npz')).todense()
    MDM_ps=sp.load_npz("{}{}".format(path, 'MDM_ps.npz')).todense()
    MWM_ps=sp.load_npz("{}{}".format(path, 'MWM_ps.npz')).todense()

    # MAM;
    MAM_emb = []
    for v in MAMi:
        result = {}
        for m in MAMi[v]:
            np1 = len(MAMi[v][m])
            edge1 = [MAM_e[p] for p in MAMi[v][m]]
            edge1 = np.sum(np.vstack(edge1), axis=0)  # edge1: the emd between v and a1
            edge1 /= np1
            edge1 += MAM_e[v] + MAM_e[m]
            if m not in result:
                result[m] = edge1
            else:
                result[m] += edge1

            if v <= m:
                MAM_emb.append(np.concatenate(([v, m], result[m]/3, [MAM_ps[v,m]], [np1])))
    MAM_emb = np.asarray(MAM_emb)
    m = np.max(MAM_emb[:,-1])
    MAM_emb[:,-1]/=m
    print("compute edge embeddings {} complete".format('MAM'))

    # MDM;
    MDM_emb = []
    for v in MDMi:
        result = {}
        for m in MDMi[v]:
            np1 = len(MDMi[v][m])
            edge1 = [MDM_e[p] for p in MDMi[v][m]]
            edge1 = np.sum(np.vstack(edge1), axis=0)  # edge1: the emd between v and a1
            edge1 /= np1
            edge1 += MDM_e[v] + MDM_e[m]
            if m not in result:
                result[m] = edge1
            else:
                result[m] += edge1

            if v <= m:
                MDM_emb.append(np.concatenate(([v, m], result[m]/3,[MDM_ps[v,m]], [np1])))
    MDM_emb = np.asarray(MDM_emb)
    m = np.max(MDM_emb[:, -1])
    MDM_emb[:, -1] /= m
    print("compute edge embeddings {} complete".format('MDM'))

    # MWM;
    MWM_emb = []
    for v in MWMi:
        result = {}
        for m in MWMi[v]:
            np1 = len(MWMi[v][m])
            edge1 = [MWM_e[p] for p in MWMi[v][m]]
            edge1 = np.sum(np.vstack(edge1), axis=0)  # edge1: the emd between v and a1
            edge1 /= np1
            edge1 += MWM_e[v] + MWM_e[m]
            if m not in result:
                result[m] = edge1
            else:
                result[m] += edge1

            if v <= m:
                MWM_emb.append(np.concatenate(([v, m], result[m]/3,[MWM_ps[v,m]], [np1])))
    MWM_emb = np.asarray(MWM_emb)
    m = np.max(MWM_emb[:, -1])
    MWM_emb[:, -1] /= m
    print("compute edge embeddings {} complete".format('MWM'))

    emb_len = MWM_emb.shape[1]-2
    np.savez("{}edge{}.npz".format(path, emb_len),
             MAM=MAM_emb, MDM=MDM_emb, MWM=MWM_emb)
    print('dump npz file {}edge{}.npz complete'.format(path, emb_len))
    pass


def gen_yago_randomwalk(path='../../../data/yago/',
                        walk_length=80,n_walks=10):
    # dump APA
    label_file = "labels"
    MA_file = "movie_actor"
    MD_file = "movie_director"
    MW_file = "movie_writer"

    movies = []
    actors = []
    directors = []
    writers = []

    with open('{}{}.txt'.format(path, "movies"), mode='r', encoding='UTF-8') as f:
        for line in f:
            movies.append(line.split()[0])

    with open('{}{}.txt'.format(path, "actors"), mode='r', encoding='UTF-8') as f:
        for line in f:
            actors.append(line.split()[0])

    with open('{}{}.txt'.format(path, "directors"), mode='r', encoding='UTF-8') as f:
        for line in f:
            directors.append(line.split()[0])

    with open('{}{}.txt'.format(path, "writers"), mode='r', encoding='UTF-8') as f:
        for line in f:
            writers.append(line.split()[0])

    n_movie = len(movies)  # 1465
    n_actor = len(actors)  # 4019
    n_director = len(directors)  # 1093
    n_writer = len(writers)  # 1458

    movie_dict = {a: i for (i, a) in enumerate(movies)}
    actor_dict = {a: i + n_movie for (i, a) in enumerate(actors)}
    director_dict = {a: i + n_movie + n_actor for (i, a) in enumerate(directors)}
    writer_dict = {a: i + n_movie + n_actor + n_director for (i, a) in enumerate(writers)}

    MA = []
    with open('{}{}.txt'.format(path, MA_file), 'r', encoding='UTF-8') as f:
        for line in f:
            arr = line.split()
            MA.append([movie_dict[arr[0]], actor_dict[arr[1]]])

    MD = []
    with open('{}{}.txt'.format(path, MD_file), 'r', encoding='UTF-8') as f:
        for line in f:
            arr = line.split()
            MD.append([movie_dict[arr[0]], director_dict[arr[1]]])

    MW = []
    with open('{}{}.txt'.format(path, MW_file), 'r', encoding='UTF-8') as f:
        for line in f:
            arr = line.split()
            MW.append([movie_dict[arr[0]], writer_dict[arr[1]]])

    MA = np.asarray(MA)
    MD = np.asarray(MD)
    MW = np.asarray(MW)

    #--
    #build index for 2hop adjs

    MAi={}
    MDi={}
    MWi={}
    AMi={}
    DMi={}
    WMi={}

    for i in range(MA.shape[0]):
        m=MA[i,0]
        a=MA[i,1]

        if m not in MAi:
            MAi[m]=set()
        if a not in AMi:
            AMi[a]=set()

        MAi[m].add(a)
        AMi[a].add(m)

    for i in range(MD.shape[0]):
        m = MD[i, 0]
        d = MD[i, 1]

        if m not in MDi:
            MDi[m] = set()
        if d not in DMi:
            DMi[d] = set()

        MDi[m].add(d)
        DMi[d].add(m)

    for i in range(MW.shape[0]):
        m = MW[i, 0]
        w = MW[i, 1]

        if m not in MWi:
            MWi[m] = set()
        if w not in WMi:
            WMi[w] = set()

        MWi[m].add(w)
        WMi[w].add(m)

    index={}
    index['AM'] = AMi
    index['DM'] = DMi
    index['WM'] = WMi
    index['MA'] = MAi
    index['MD'] = MDi
    index['MW'] = MWi

    schemes=["MWM","MAM","MDM"]

    for scheme in schemes:
        ind1 = index[scheme[:2]]
        ind2 = index[scheme[1:]]
        with open('{}{}.walk'.format(path,scheme),'w') as f:

            for v in ind1:

                for n in range(n_walks):
                    out="a{}".format(v)

                    m = v
                    for w in range(int(walk_length/2)):
                        a = np.random.choice(tuple(ind1[m]))
                        out += " v{}".format(a)
                        m = np.random.choice(tuple(ind2[a]))
                        out += " a{}".format(m)

                    f.write(out+"\n")
            pass
        print('file {}.walk dumped'.format(scheme))
        pass


def pathsim(A):
    value = []
    x,y = A.nonzero()
    for i,j in zip(x,y):
        value.append(2 * A[i, j] / (A[i, i] + A[j, j]))
    return sp.coo_matrix((value,(x,y)))

def gen_homoadj(path):
    label_file = "labels"
    MA_file = "movie_actor"
    MD_file = "movie_director"
    MW_file = "movie_writer"

    movies = []
    actors = []
    directors = []
    writers = []

    with open('{}{}.txt'.format(path, "movies"), mode='r', encoding='UTF-8') as f:
        for line in f:
            movies.append(line.split()[0])

    with open('{}{}.txt'.format(path, "actors"), mode='r', encoding='UTF-8') as f:
        for line in f:
            actors.append(line.split()[0])

    with open('{}{}.txt'.format(path, "directors"), mode='r', encoding='UTF-8') as f:
        for line in f:
            directors.append(line.split()[0])

    with open('{}{}.txt'.format(path, "writers"), mode='r', encoding='UTF-8') as f:
        for line in f:
            writers.append(line.split()[0])

    n_movie = len(movies)     #1465
    n_actor = len(actors)    #4019
    n_director = len(directors) #1093
    n_writer = len(writers)    #1458

    movie_dict = {a: i for (i, a) in enumerate(movies)}
    actor_dict = {a: i for (i, a) in enumerate(actors)}
    director_dict = {a: i for (i, a) in enumerate(directors)}
    writer_dict = {a: i for (i, a) in enumerate(writers)}


    MA = []
    with open('{}{}.txt'.format(path, MA_file), 'r', encoding='UTF-8') as f:
        for line in f:
            arr = line.split()
            MA.append([movie_dict[arr[0]], actor_dict[arr[1]] ])

    MD = []
    with open('{}{}.txt'.format(path, MD_file), 'r', encoding='UTF-8') as f:
        for line in f:
            arr = line.split()
            MD.append([movie_dict[arr[0]], director_dict[arr[1]]])

    MW = []
    with open('{}{}.txt'.format(path, MW_file), 'r', encoding='UTF-8') as f:
        for line in f:
            arr = line.split()
            MW.append([movie_dict[arr[0]], writer_dict[arr[1]]])

    MA = np.asarray(MA)
    MD = np.asarray(MD)
    MW = np.asarray(MW)

    MA = sp.coo_matrix((np.ones(MA.shape[0]), (MA[:, 0], MA[:, 1])),
                       shape=(n_movie, n_actor),
                       dtype=np.float32)
    MD = sp.coo_matrix((np.ones(MD.shape[0]), (MD[:, 0], MD[:, 1])),
                       shape=(n_movie, n_director),
                       dtype=np.float32)
    MW = sp.coo_matrix((np.ones(MW.shape[0]), (MW[:, 0], MW[:, 1])),
                       shape=(n_movie, n_writer),
                       dtype=np.float32)

    MAM = MA * MA.transpose()
    MDM = MD * MD.transpose()
    MWM = MW * MW.transpose()

    #MAM = pathsim(MAM)
    #MDM = pathsim(MDM)
    #MWM = pathsim(MWM)

    #sp.save_npz("{}{}".format(path, 'MAM_ps.npz'), MAM)
    #sp.save_npz("{}{}".format(path, 'MDM_ps.npz'), MDM)
    #sp.save_npz("{}{}".format(path, 'MWM_ps.npz'), MWM)

    MAM = np.hstack([MAM.nonzero()[0].reshape(-1,1), MAM.nonzero()[1].reshape(-1,1)])
    MDM = np.hstack([MDM.nonzero()[0].reshape(-1,1), MDM.nonzero()[1].reshape(-1,1)])
    MWM = np.hstack([MWM.nonzero()[0].reshape(-1,1), MWM.nonzero()[1].reshape(-1,1)])

    
    np.savetxt("{}{}.txt".format(path, 'MAM'),MAM,fmt='%u')
    np.savetxt("{}{}.txt".format(path, 'MDM'),MDM,fmt='%u')
    np.savetxt("{}{}.txt".format(path, 'MWM'),MWM,fmt='%u')


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


def gen_edge_adj(path='data/freebase/', K=80,edge_dim=130,ps=True):
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

    MA_file = "movie_actor"
    MD_file = "movie_director"
    MW_file = "movie_writer"

    movies = []
    actors = []
    directors = []
    writers = []

    with open('{}{}.txt'.format(path, "movies"), mode='r', encoding='UTF-8') as f:
        for line in f:
            movies.append(line.split()[0])

    with open('{}{}.txt'.format(path, "actors"), mode='r', encoding='UTF-8') as f:
        for line in f:
            actors.append(line.split()[0])

    with open('{}{}.txt'.format(path, "directors"), mode='r', encoding='UTF-8') as f:
        for line in f:
            directors.append(line.split()[0])

    with open('{}{}.txt'.format(path, "writers"), mode='r', encoding='UTF-8') as f:
        for line in f:
            writers.append(line.split()[0])

    n_movie = len(movies)  # 1465
    n_actor = len(actors)  # 4019
    n_director = len(directors)  # 1093
    n_writer = len(writers)  # 1458

    movie_dict = {a: i for (i, a) in enumerate(movies)}
    actor_dict = {a: i for (i, a) in enumerate(actors)}
    director_dict = {a: i for (i, a) in enumerate(directors)}
    writer_dict = {a: i for (i, a) in enumerate(writers)}

    MA = []
    with open('{}{}.txt'.format(path, MA_file), 'r', encoding='UTF-8') as f:
        for line in f:
            arr = line.split()
            MA.append([movie_dict[arr[0]], actor_dict[arr[1]]])

    MD = []
    with open('{}{}.txt'.format(path, MD_file), 'r', encoding='UTF-8') as f:
        for line in f:
            arr = line.split()
            MD.append([movie_dict[arr[0]], director_dict[arr[1]]])

    MW = []
    with open('{}{}.txt'.format(path, MW_file), 'r', encoding='UTF-8') as f:
        for line in f:
            arr = line.split()
            MW.append([movie_dict[arr[0]], writer_dict[arr[1]]])

    MA = np.asarray(MA)
    MD = np.asarray(MD)
    MW = np.asarray(MW)

    MA = sp.coo_matrix((np.ones(MA.shape[0]), (MA[:, 0], MA[:, 1])),
                       shape=(n_movie, n_actor),
                       dtype=np.float32)
    MD = sp.coo_matrix((np.ones(MD.shape[0]), (MD[:, 0], MD[:, 1])),
                       shape=(n_movie, n_director),
                       dtype=np.float32)
    MW = sp.coo_matrix((np.ones(MW.shape[0]), (MW[:, 0], MW[:, 1])),
                       shape=(n_movie, n_writer),
                       dtype=np.float32)

    if ps:
        MAM_ps = sparse_mx_to_torch_sparse_tensor(sp.load_npz("{}{}".format(path, 'MAM_ps.npz'))).to_dense()
        MDM_ps = sparse_mx_to_torch_sparse_tensor(sp.load_npz("{}{}".format(path, 'MDM_ps.npz'))).to_dense()
        MWM_ps = sparse_mx_to_torch_sparse_tensor(sp.load_npz("{}{}".format(path, 'MWM_ps.npz'))).to_dense()
        adj = {'MAM': MAM_ps, 'MDM': MDM_ps, 'MWM': MWM_ps}
    else:
        MAM = sparse_mx_to_torch_sparse_tensor(MA * MA.transpose()).to_dense().long()
        MDM = sparse_mx_to_torch_sparse_tensor(MD * MD.transpose()).to_dense().long()
        MWM = sparse_mx_to_torch_sparse_tensor(MW * MW.transpose()).to_dense().long()
        adj = {'MAM': MAM, 'MDM': MDM, 'MWM': MWM}
    #select top-K path-count neighbors of each A. If number of neighbors>K, trunc; else upsampling

    schemes = ['MAM','MDM','MWM']#
    index, emb=load_edge_emb(path, schemes, n_dim=edge_dim, n_author=n_movie)

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

        print('edge_emb.shape ', edge_emb.shape)

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
        # node_neighs[s] = node_neigh.numpy()
        node2edge_idxs[s] = edge_idx.numpy()
        edge_embs[s] =edge_emb.numpy()
        # print(edge_embs[s].shape)
        # edge2node_idxs[s] = edge2node
        edge_node_adjs[s] = edge_node_adj

    # np.savez("{}edge_neighs_{}_{}.npz".format(path,K,out_dims),
    #          MAM=edge_neighs['MAM'], MDM=edge_neighs['MDM'], MWM=edge_neighs['MWM'],)
    # print('dump npz file {}edge_neighs.npz complete'.format(path))

    # np.savez("{}node_neighs_{}_{}.npz".format(path,K,out_dims),
    #          MAM=node_neighs['MAM'], MDM=node_neighs['MDM'], MWM=node_neighs['MWM'],)
    # print('dump npz file {}node_neighs.npz complete'.format(path))

    np.savez("{}node2edge_idxs_{}_{}.npz".format(path,K,out_dims),
             MAM=node2edge_idxs['MAM'], MDM=node2edge_idxs['MDM'], MWM=node2edge_idxs['MWM'],)
    print('dump npz file {}node2edge_idxs.npz complete'.format(path))

    np.savez("{}edge_embs_{}_{}.npz".format(path,K,out_dims),
             MAM=edge_embs['MAM'], MDM=edge_embs['MDM'], MWM=edge_embs['MWM'],)
    print('dump npz file {}edge_embs.npz complete'.format(path))

    # np.savez("{}edge2node_idxs_{}_{}.npz".format(path,K,out_dims),
    #          MAM=edge2node_idxs['MAM'], MDM=edge2node_idxs['MDM'], MWM=edge2node_idxs['MWM'],)
    # print('dump npz file {}edge2node_idxs.npz complete'.format(path))

    np.savez("{}edge_node_adjs_{}_{}.npz".format(path, K,out_dims),
             MAM=edge_node_adjs['MAM'], MDM=edge_node_adjs['MDM'], MWM=edge_node_adjs['MWM'],)
    print('dump npz file {}edge_node_adjs.npz complete'.format(path))

    pass


def gen_edge_adj_random(path='data/freebase/', edge_dim=130,ps=True):
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

    MA_file = "movie_actor"
    MD_file = "movie_director"
    MW_file = "movie_writer"

    movies = []
    actors = []
    directors = []
    writers = []

    with open('{}{}.txt'.format(path, "movies"), mode='r', encoding='UTF-8') as f:
        for line in f:
            movies.append(line.split()[0])

    with open('{}{}.txt'.format(path, "actors"), mode='r', encoding='UTF-8') as f:
        for line in f:
            actors.append(line.split()[0])

    with open('{}{}.txt'.format(path, "directors"), mode='r', encoding='UTF-8') as f:
        for line in f:
            directors.append(line.split()[0])

    with open('{}{}.txt'.format(path, "writers"), mode='r', encoding='UTF-8') as f:
        for line in f:
            writers.append(line.split()[0])

    n_movie = len(movies)  # 1465
    n_actor = len(actors)  # 4019
    n_director = len(directors)  # 1093
    n_writer = len(writers)  # 1458

    movie_dict = {a: i for (i, a) in enumerate(movies)}
    actor_dict = {a: i for (i, a) in enumerate(actors)}
    director_dict = {a: i for (i, a) in enumerate(directors)}
    writer_dict = {a: i for (i, a) in enumerate(writers)}

    MA = []
    with open('{}{}.txt'.format(path, MA_file), 'r', encoding='UTF-8') as f:
        for line in f:
            arr = line.split()
            MA.append([movie_dict[arr[0]], actor_dict[arr[1]]])

    MD = []
    with open('{}{}.txt'.format(path, MD_file), 'r', encoding='UTF-8') as f:
        for line in f:
            arr = line.split()
            MD.append([movie_dict[arr[0]], director_dict[arr[1]]])

    MW = []
    with open('{}{}.txt'.format(path, MW_file), 'r', encoding='UTF-8') as f:
        for line in f:
            arr = line.split()
            MW.append([movie_dict[arr[0]], writer_dict[arr[1]]])

    MA = np.asarray(MA)
    MD = np.asarray(MD)
    MW = np.asarray(MW)

    MA = sp.coo_matrix((np.ones(MA.shape[0]), (MA[:, 0], MA[:, 1])),
                       shape=(n_movie, n_actor),
                       dtype=np.float32)
    MD = sp.coo_matrix((np.ones(MD.shape[0]), (MD[:, 0], MD[:, 1])),
                       shape=(n_movie, n_director),
                       dtype=np.float32)
    MW = sp.coo_matrix((np.ones(MW.shape[0]), (MW[:, 0], MW[:, 1])),
                       shape=(n_movie, n_writer),
                       dtype=np.float32)

    if ps:
        MAM_ps = sparse_mx_to_torch_sparse_tensor(sp.load_npz("{}{}".format(path, 'MAM_ps.npz'))).to_dense()
        MDM_ps = sparse_mx_to_torch_sparse_tensor(sp.load_npz("{}{}".format(path, 'MDM_ps.npz'))).to_dense()
        MWM_ps = sparse_mx_to_torch_sparse_tensor(sp.load_npz("{}{}".format(path, 'MWM_ps.npz'))).to_dense()
        adj = {'MAM': MAM_ps, 'MDM': MDM_ps, 'MWM': MWM_ps}
    else:
        MAM = sparse_mx_to_torch_sparse_tensor(MA * MA.transpose()).to_dense().long()
        MDM = sparse_mx_to_torch_sparse_tensor(MD * MD.transpose()).to_dense().long()
        MWM = sparse_mx_to_torch_sparse_tensor(MW * MW.transpose()).to_dense().long()
        adj = {'MAM': MAM, 'MDM': MDM, 'MWM': MWM}
    #select top-K path-count neighbors of each A. If number of neighbors>K, trunc; else upsampling

    schemes = ['MAM','MDM','MWM']#
    index, emb=load_edge_emb(path, schemes, n_dim=edge_dim, n_author=n_movie)

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

        print('edge_emb.shape ', edge_emb.shape)

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
        # node_neighs[s] = node_neigh.numpy()
        node2edge_idxs[s] = edge_idx.numpy()
        edge_embs[s] =edge_emb.numpy()
        # print(edge_embs[s].shape)
        # edge2node_idxs[s] = edge2node
        edge_node_adjs[s] = edge_node_adj

    # np.savez("{}edge_neighs_{}_{}.npz".format(path,K,out_dims),
    #          MAM=edge_neighs['MAM'], MDM=edge_neighs['MDM'], MWM=edge_neighs['MWM'],)
    # print('dump npz file {}edge_neighs.npz complete'.format(path))

    # np.savez("{}node_neighs_{}_{}.npz".format(path,K,out_dims),
    #          MAM=node_neighs['MAM'], MDM=node_neighs['MDM'], MWM=node_neighs['MWM'],)
    # print('dump npz file {}node_neighs.npz complete'.format(path))

    np.savez("{}node2edge_idxs_{}_{}.npz".format(path,K,out_dims),
             MAM=node2edge_idxs['MAM'], MDM=node2edge_idxs['MDM'], MWM=node2edge_idxs['MWM'],)
    print('dump npz file {}node2edge_idxs.npz complete'.format(path))

    np.savez("{}edge_embs_{}_{}.npz".format(path,K,out_dims),
             MAM=edge_embs['MAM'], MDM=edge_embs['MDM'], MWM=edge_embs['MWM'],)
    print('dump npz file {}edge_embs.npz complete'.format(path))

    # np.savez("{}edge2node_idxs_{}_{}.npz".format(path,K,out_dims),
    #          MAM=edge2node_idxs['MAM'], MDM=edge2node_idxs['MDM'], MWM=edge2node_idxs['MWM'],)
    # print('dump npz file {}edge2node_idxs.npz complete'.format(path))

    np.savez("{}edge_node_adjs_{}_{}.npz".format(path, K,out_dims),
             MAM=edge_node_adjs['MAM'], MDM=edge_node_adjs['MDM'], MWM=edge_node_adjs['MWM'],)
    print('dump npz file {}edge_node_adjs.npz complete'.format(path))

    pass


def gen_edge_sim_adj(path='data/freebase/', K=80,edge_dim=66,sim='cos'):
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
    MAM_e, n_nodes, emb_len = read_embed(path=path,emb_file="MAM_{}".format(edge_dim-2))
    MDM_e, n_nodes, emb_len = read_embed(path=path,emb_file="MDM_{}".format(edge_dim-2))
    MWM_e, n_nodes, emb_len = read_embed(path=path,emb_file="MWM_{}".format(edge_dim-2))
    
    MA_file = "movie_actor"
    MD_file = "movie_director"
    MW_file = "movie_writer"

    movies = []
    actors = []
    directors = []
    writers = []

    with open('{}{}.txt'.format(path, "movies"), mode='r', encoding='UTF-8') as f:
        for line in f:
            movies.append(line.split()[0])

    with open('{}{}.txt'.format(path, "actors"), mode='r', encoding='UTF-8') as f:
        for line in f:
            actors.append(line.split()[0])

    with open('{}{}.txt'.format(path, "directors"), mode='r', encoding='UTF-8') as f:
        for line in f:
            directors.append(line.split()[0])

    with open('{}{}.txt'.format(path, "writers"), mode='r', encoding='UTF-8') as f:
        for line in f:
            writers.append(line.split()[0])

    n_movie = len(movies)  # 1465
    n_actor = len(actors)  # 4019
    n_director = len(directors)  # 1093
    n_writer = len(writers)  # 1458

    movie_dict = {a: i for (i, a) in enumerate(movies)}
    actor_dict = {a: i for (i, a) in enumerate(actors)}
    director_dict = {a: i for (i, a) in enumerate(directors)}
    writer_dict = {a: i for (i, a) in enumerate(writers)}

    MA = []
    with open('{}{}.txt'.format(path, MA_file), 'r', encoding='UTF-8') as f:
        for line in f:
            arr = line.split()
            MA.append([movie_dict[arr[0]], actor_dict[arr[1]]])

    MD = []
    with open('{}{}.txt'.format(path, MD_file), 'r', encoding='UTF-8') as f:
        for line in f:
            arr = line.split()
            MD.append([movie_dict[arr[0]], director_dict[arr[1]]])

    MW = []
    with open('{}{}.txt'.format(path, MW_file), 'r', encoding='UTF-8') as f:
        for line in f:
            arr = line.split()
            MW.append([movie_dict[arr[0]], writer_dict[arr[1]]])

    MA = np.asarray(MA)
    MD = np.asarray(MD)
    MW = np.asarray(MW)

    MA = sp.coo_matrix((np.ones(MA.shape[0]), (MA[:, 0], MA[:, 1])),
                       shape=(n_movie, n_actor),
                       dtype=np.float32)
    MD = sp.coo_matrix((np.ones(MD.shape[0]), (MD[:, 0], MD[:, 1])),
                       shape=(n_movie, n_director),
                       dtype=np.float32)
    MW = sp.coo_matrix((np.ones(MW.shape[0]), (MW[:, 0], MW[:, 1])),
                       shape=(n_movie, n_writer),
                       dtype=np.float32)

    if sim=='cos':
        from sklearn.metrics.pairwise import cosine_similarity
        MAM = torch.from_numpy(cosine_similarity(MA * MA.transpose()))
        MDM = torch.from_numpy(cosine_similarity(MD * MD.transpose()))
        MWM = torch.from_numpy(cosine_similarity(MW * MW.transpose()))
        # print(MAM[0])
        adj = {'MAM': MAM, 'MDM': MDM, 'MWM': MWM}
    else:
        MAM = sparse_mx_to_torch_sparse_tensor(MA * MA.transpose()).to_dense().long()
        MDM = sparse_mx_to_torch_sparse_tensor(MD * MD.transpose()).to_dense().long()
        MWM = sparse_mx_to_torch_sparse_tensor(MW * MW.transpose()).to_dense().long()
        adj = {'MAM': MAM, 'MDM': MDM, 'MWM': MWM}
    #select top-K path-count neighbors of each A. If number of neighbors>K, trunc; else upsampling

    schemes = ['MAM','MDM','MWM']#
    index, emb=load_edge_emb(path, schemes, n_dim=edge_dim, n_author=n_movie)
    node_emb={'MAM':MAM_e,'MDM':MDM_e,'MWM':MWM_e,}

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
            # edge_hash2 = n2*node_neigh.shape[0]+n1
            
            if edge_hash1 in edgeHash2emb:
                edge_idx_new.append(edgeHash2emb[edge_hash1])
            else:
                edgeHash2emb[edge_hash1] = len(new_embs)
                # edgeHash2emb[edge_hash2] = len(new_embs)
                
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
             MAM=node2edge_idxs['MAM'], MDM=node2edge_idxs['MDM'], MWM=node2edge_idxs['MWM'],)
    print('dump npz file {}node2edge_idxs.npz complete'.format(path))

    np.savez("{}edge_embs_{}_{}_cos.npz".format(path,K,out_dims),
             MAM=edge_embs['MAM'], MDM=edge_embs['MDM'], MWM=edge_embs['MWM'],)
    print('dump npz file {}edge_embs.npz complete'.format(path))

    np.savez("{}edge_node_adjs_{}_{}_cos.npz".format(path, K,out_dims),
             MAM=edge_node_adjs['MAM'], MDM=edge_node_adjs['MDM'], MWM=edge_node_adjs['MWM'],)
    print('dump npz file {}edge_node_adjs.npz complete'.format(path))

    pass


if __name__ == '__main__':
    # gen_homograph(path='../../../data/freebase/')

    # dump_yago_edge_emb(path='../data/freebase/',edge_len=128)

    # gen_yago_randomwalk(path='../data/freebase/',
    #             walk_length=100,n_walks=1000)

    #gen_homoadj(path='data/freebase/')

    #gen_edge_adj(path='data/freebase/', K=10,edge_dim=130)

    gen_edge_adj(path='data/freebase/', K=5, edge_dim=34)
