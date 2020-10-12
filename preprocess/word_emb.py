import numpy as np
import os
import sys
# sys.path.append('.')
import pickle

def read_emb(terms='../data/dblp3/term_name.txt', emb='../data/glove.840B.300d.txt', out_file='../data/dblp3/term_emb.npz'):
    '''

    Args:
        terms:
        emb:

    Returns:
        termid2emb: term to embed mapping, term id starts from 0.
    '''
    term_list = []
    with open(terms,'r') as f:

        for line in f:
            id,term = line.strip().split()
            term_list.append(term)

    print(len(term_list))

    term_dict = dict()
    with open(emb,'r',encoding="utf-8") as f:
        for line in f:
            line = line.strip().split()
            term = line[0]
            if term not in term_list:
                continue
            try:
                term_dict[term] = [float(line[i]) for i in range(1,len(line))]
            except ValueError:
                print(len(line), line)
    print(len(term_dict))
    embs=np.asarray([ term_dict[term] if term in term_dict else np.zeros(300) for term in term_list ])
    print('embs.shape ',embs.shape )

    # if not os.path.isfile(out_file):
        #dump embedding
        # f = open(out_file, "wb")
        # pickle.dump(term_dict, f)
        # f.close()

    np.save(out_file,embs)

        # pass

    pass


def count2feat(feat, emb):
    import scipy.sparse as sp
    rowsum = np.array(feat.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)

    print(feat.shape, emb.shape)

    feat = feat.dot(emb)
    feat = r_mat_inv.dot(feat)

    return feat


if __name__ =='__main__':
    read_emb(terms='../data/dblp2/term_name.txt', out_file='../data/dblp2/term_emb.npy')
    pass