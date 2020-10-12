### Conch
A semisupervised learning model for heterogeneous information networks (HINs)

#### Dependency
ujson == 1.35
pytorch >= 1.5.0

#### Preprocess
0. (DBLP dataset only) Download Glove word embdding: http://nlp.stanford.edu/data/glove.840B.300d.zip
1. Generate node embeddings by node2vec or metapath2vec:
    a.    preprocess/yelp.py -> gen_homograph()
          preprocess/main.py --input ../../data/yelp/homograph.txt --output ../../data/ yelp/RUBK_128.emb --dimensions 128 --workers 56 --walk-length 100            --num-walks 40 --window-size 5
    b.    preprocess/yelp.py -> gen_walk(path='../data/yelp/',                                      walk_length=100,n_walks=40)
          preprocess/metapath2vec -train ../../data/yelp/BRKRB.walk -output ../../data/yelp/BRKRB_128.emb -size 128 -threads 40

2. Fuse edge features:
    preprocess/yelp.py -> dump_yelp_edge_emb(path='../data/yelp/')

3. Compute index:
    preprocess/yelp.py -> gen_edge_adj_random(path='../data/yelp/',edge_dim=130)

#### Run
# Driver for Multiple runs: 
run_dblp.py; run_freebase.py; run_yelp.py

# Entry for training program:
Conch: train_reg3.py
Conch_nc: train_ncreg.py
Conch_rd: train_rdreg.py

# Entry for training programs without contrastive learning regularization:
Conch: train.py
Conch_nc: train_nc.py
Conch_rd: train_rd.py

##### LICENSE
MIT


