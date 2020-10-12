#!/bin/bash

./metapath2vec -train ../../data/freebase/MAM.walk -output ../../data/freebase/MAM_64.emb -size 64 -threads 40 
./metapath2vec -train ../../data/freebase/MDM.walk -output ../../data/freebase/MDM_64.emb -size 64 -threads 40 
./metapath2vec -train ../../data/freebase/MWM.walk -output ../../data/freebase/MWM_64.emb -size 64 -threads 40 

./metapath2vec -train ../../data/freebase/MAM.walk -output ../../data/freebase/MAM_32.emb -size 32 -threads 40 
./metapath2vec -train ../../data/freebase/MDM.walk -output ../../data/freebase/MDM_32.emb -size 32 -threads 40 
./metapath2vec -train ../../data/freebase/MWM.walk -output ../../data/freebase/MWM_32.emb -size 32 -threads 40 

./metapath2vec -train ../../data/freebase/MAM.walk -output ../../data/freebase/MAM_256.emb -size 256 -threads 40 
./metapath2vec -train ../../data/freebase/MDM.walk -output ../../data/freebase/MDM_256.emb -size 256 -threads 40 
./metapath2vec -train ../../data/freebase/MWM.walk -output ../../data/freebase/MWM_256.emb -size 256 -threads 40 
