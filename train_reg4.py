#!/usr/bin/env python

"""
    train.py
"""

from __future__ import division
from __future__ import print_function
import os
from functools import partial
import sys
import argparse
import ujson as json
import numpy as np
from time import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.autograd import Variable
from torch.nn import functional as F

from models import HINGCN_GS, MyDataParallel,HINGCN_Dense
from problem import NodeProblem, ReadCosSim
from helpers import set_seeds, to_numpy
from nn_modules import aggregator_lookup, prep_lookup, sampler_lookup, edge_aggregator_lookup, \
    metapath_aggregator_lookup
from lr import LRSchedule
from model.models import *
import math
# --
# Helpers

def set_progress(optimizer, lr_scheduler, progress):
    lr = lr_scheduler(progress)
    LRSchedule.set_lr(optimizer, lr)

def evaluate_unsupervised(model, problem, batch_size, loss_fn,):
    pred = model(problem.feats, feat2=None, msk=None, samp_bias1=None, samp_bias2=None, get_embed='embed')
    pred = pred.detach().cpu().numpy()
    for ids, targets, epoch_progress in problem.iterate(mode='train', shuffle=True, batch_size=999999):
        x_train = pred[ids.cpu()]
        y_train = targets.cpu().numpy().ravel()
    for ids, targets, epoch_progress in problem.iterate(mode='val', shuffle=True, batch_size=999999):
        x_val = pred[ids.cpu()]
        y_val = targets.cpu().numpy().ravel()
    for ids, targets, epoch_progress in problem.iterate(mode='test', shuffle=True, batch_size=999999):
        x_test = pred[ids.cpu()]
        y_test = targets.cpu().numpy().ravel()
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    # fpr, tpr, thresholds = metrics.roc_curve(y_test, y_scores, pos_label=2)
    # plot_roc_curve(fpr,tpr,'ROC')
    y_pred=logreg.predict(x_test)
    y_train_scores = logreg.predict_proba(x_train)[:, 1]
    y_test_scores = logreg.predict_proba(x_test)[:, 1]
    # print('Accuracy of Logistic regression classifier on training set: {:.4f}'
    #       .format(logreg.score(x_train, y_train)))
    # print('Accuracy of Logistic regression classifier on test set: {:.4f}'
    #       .format(logreg.score(x_test, y_test)))
    # print('Macro of Logistic regression classifier on test set: {:.4f}'
    #       .format(f1_score(y_test,y_pred, average='macro')))
    # print('Micro of Logistic regression classifier on test set: {:.4f}'
    #       .format(f1_score(y_test,y_pred, average='micro')))

    result = json.dumps({
                "test_metric": {'accuracy':f1_score(y_test,y_pred, average='micro'),
                'macro':f1_score(y_test,y_pred, average='macro')
                },
            }, double_precision=5)
    return result


def evaluate(model, problem, batch_size, loss_fn, coff, mode='val'):
    assert mode in ['test', 'val']
    preds, acts = [], []
    loss=0
    for (ids, targets, _) in problem.iterate(mode=mode, shuffle=False, batch_size=batch_size):
        # print(ids.shape,targets.shape)
        pred = model(problem.feats, feat2=None, msk=None, samp_bias1=None, samp_bias2=None, get_embed='pred')
        # pred = model.get_embed(problem.feats)
        loss += loss_fn(pred[ids], targets.squeeze()).item()
        preds.append(to_numpy(pred[ids]))
        acts.append(to_numpy(targets))
    #
    return loss, problem.metric_fn(np.vstack(acts), np.vstack(preds))

# # --
# Args

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--problem-path', type=str, default='data/dblp2/')
    parser.add_argument('--problem', type=str, default='dblp')

    parser.add_argument('--no-cuda', action="store_true",default=False)

    # Optimization params
    parser.add_argument('--batch-size', type=int, default=99999)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr-init', type=float, default=0.001)
    parser.add_argument('--lr-schedule', type=str, default='constant')
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batchnorm', action="store_true")
    parser.add_argument('--tolerance', type=int, default=100)
    parser.add_argument('--attn-dropout',type=float,default=0)
    # Architecture params
    parser.add_argument('--sampler-class', type=str, default='sparse_uniform_neighbor_sampler')

    parser.add_argument('--prep-class', type=str, default='linear')  # linear,identity,node_embedding
    parser.add_argument('--prep-len', type=int, default=256)
    parser.add_argument('--in-edge-len', type=int, default=128)
    parser.add_argument('--aggregator-class', type=str, default='sum')
    parser.add_argument('--edge-aggr-class', type=str, default='sum')
    parser.add_argument('--mpaggr-class', type=str, default='attention')

    parser.add_argument('--concat-node', action="store_true",default=False)
    parser.add_argument('--concat-edge', action="store_true")

    parser.add_argument('--n-head', type=int, default=1)
    parser.add_argument('--k', type=int, default=10)
    # parser.add_argument('--n-train-samples', type=str, default='600,600')
    # parser.add_argument('--n-val-samples', type=str, default='600,600')
    parser.add_argument('--output-dims', type=str, default='128,128,32,32')
    parser.add_argument('--n-layer', type=int, default='2')
    
    parser.add_argument('--train-per', type=float, default=0.1)
    parser.add_argument('--coff-scheme', type=str, default='exp')
    parser.add_argument('--max-coff', type=float, default=1)
    parser.add_argument('--max-epoch', type=float, default=100)
    parser.add_argument('--coff-exp', type=float, default=5)

    # Logging
    parser.add_argument('--log-interval', default=1, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--show-test', action="store_true")

    # --
    # Validate args

    args = parser.parse_args()
    args.cuda = not args.no_cuda
    assert args.prep_class in prep_lookup.keys(), 'parse_args: prep_class not in %s' % str(prep_lookup.keys())
    assert args.aggregator_class in aggregator_lookup.keys(), 'parse_args: aggregator_class not in %s' % str(
        aggregator_lookup.keys())
    assert args.batch_size > 1, 'parse_args: batch_size must be > 1'
    return args


if __name__ == "__main__":
    args = parse_args()
    set_seeds(args.seed)
    verbose = args.verbose
    torch.backends.cudnn.enabled = True
    torch.cuda.set_device(0)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # --
    # Load problem
    mp_index = {'dblp':['APA','APAPA','APCPA'],#
                'yelp': [ 'BRKRB','BRURB'], #'BRURB',
                'yago': ['MAM','MDM','MWM'],
                'cora': ['PAP','PPP','PP']
                }
    schemes = mp_index[args.problem]
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    problem = NodeProblem(problem_path=args.problem_path, problem=args.problem, device=device, schemes=schemes, K=args.k, input_edge_dims =args.in_edge_len,train_per=args.train_per)
    # cos_problem = ReadCosSim(problem_path=args.problem_path, problem=args.problem, device=device, schemes=schemes, K=args.k, input_edge_dims =args.in_edge_len,train_per=args.train_per)
    # --
    # Define model

    # n_train_samples = list(map(int, args.n_train_samples.split(',')))
    # n_val_samples = list(map(int, args.n_val_samples.split(',')))
    output_dims = list(map(int, args.output_dims.split(',')))
    model = conch_dgi2(**{
        "shuffle" : False,
        "problem": problem,
        # "cos_problem": cos_problem,
        "n_mp": len(schemes),
        "sampler_class": sampler_lookup[args.sampler_class],

        "prep_class": prep_lookup[args.prep_class],
        "prep_len": args.prep_len,
        "aggregator_class": aggregator_lookup[args.aggregator_class],
        "mpaggr_class": metapath_aggregator_lookup[args.mpaggr_class],
        "edge_aggr_class": aggregator_lookup[args.edge_aggr_class],
        "n_head": args.n_head,
        "node_layer_specs": [
            {
                # "n_train_samples": n_train_samples[0],
                # "n_val_samples": n_val_samples[0],
                "output_dim": output_dims[0],
                "activation": F.relu,
                "concat_node": args.concat_node,
                "concat_edge": args.concat_edge,
            },
            {
                 # "n_train_samples": n_train_samples[1],
                 # "n_val_samples": n_val_samples[1],
                 "output_dim": output_dims[1],
                 "activation": F.relu,  # lambda x: x
                 "concat_node": args.concat_node,
                 "concat_edge": args.concat_edge,
            },
            {
                # "n_train_samples": n_train_samples[1],
                # "n_val_samples": n_val_samples[1],
                "output_dim": output_dims[2],
                "activation": F.relu,  # lambda x: x
                "concat_node": args.concat_node,
                "concat_edge": args.concat_edge,
            },
            {
                # "n_train_samples": n_train_samples[2],
                # "n_val_samples": n_val_samples[2],
                "output_dim": output_dims[3],
                "activation": F.relu,
                "concat_node": args.concat_node,
                "concat_edge": args.concat_edge,
            },
        ][:args.n_layer],
        "edge_layer_specs": [
            {
                # "n_train_samples": n_train_samples[0],
                # "n_val_samples": n_val_samples[0],
                "output_dim": output_dims[0],
                "activation": F.relu,
                "concat_node": args.concat_node,
                "concat_edge": args.concat_edge,
            },
            {
                 # "n_train_samples": n_train_samples[1],
                 # "n_val_samples": n_val_samples[1],
                 "output_dim": output_dims[1],
                 "activation": F.relu,  # lambda x: x
                 "concat_node": args.concat_node,
                 "concat_edge": args.concat_edge,
            },
            {
                # "n_train_samples": n_train_samples[1],
                # "n_val_samples": n_val_samples[1],
                "output_dim": output_dims[2],
                "activation": F.relu,  # lambda x: x
                "concat_node": args.concat_node,
                "concat_edge": args.concat_edge,
            },
            {
                # "n_train_samples": n_train_samples[2],
                # "n_val_samples": n_val_samples[2],
                "output_dim": output_dims[3],
                "activation": F.relu,
                "concat_node": args.concat_node,
                "concat_edge": args.concat_edge,
            },
        ][:args.n_layer],
        "dropout": args.dropout,
        "batchnorm": args.batchnorm,
        "attn_dropout":args.attn_dropout,
    })

    if args.cuda:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
        # model = model.cuda()
        model = model.to(device)

    # --
    # Define optimizer
    lr_scheduler = partial(getattr(LRSchedule, args.lr_schedule), lr_init=args.lr_init)
    lr = lr_scheduler(0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    # --
    # Train

    set_seeds(args.seed)

    start_time = time()
    tolerance = 0
    best_train_loss=100000
    coff = 0
    best_model = None
    #unsupervised learning
    for epoch in range(args.epochs):#args.epochs

        # early stopping
        if tolerance > args.tolerance:
            break
        train_loss = 0
        
        X = problem.feats
        idx = np.random.permutation(X.shape[0])
        X_tilda = X[idx, :]
        
        n_mp = 1
        lbl_1 = torch.ones(n_mp, X.shape[0])
        lbl_2 = torch.zeros(n_mp, X.shape[0])
        lbl = torch.cat((lbl_1, lbl_2), 1)
        if torch.cuda.is_available():
            X = X.cuda()
            X_tilda = X_tilda.cuda()
            lbl = lbl.cuda()
        
        bce = torch.nn.BCEWithLogitsLoss()
        # Train
        _ = model.train()
        for ids, targets, epoch_progress in problem.iterate(mode='train', shuffle=True, batch_size=args.batch_size):

            optimizer.zero_grad()
            preds,weights,reg = model(X, X_tilda, None, None, None, get_embed=False)
            
            if weights is not None:
                weights=weights.cpu().detach().numpy()
                if len(weights.shape)>1 and weights.shape[0] != 1:
                    weights=np.sum(weights,axis=0)/weights.shape[0]
                # print(weights)

            loss = bce(reg,lbl)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            train_loss += loss.item()
            # train_metric = problem.metric_fn(to_numpy(targets), to_numpy(preds[ids]))
        if verbose >=2:
            
            if epoch % args.log_interval == 0:
                print(json.dumps({
                "epoch": epoch,
                "time": time() - start_time,
                "train_loss": train_loss,
                "tolerance": tolerance,
                }, double_precision=5))
                sys.stdout.flush()
                test_metric =evaluate_unsupervised(model, problem, batch_size=args.batch_size,loss_fn=problem.loss_fn)
                
                print(test_metric)
        if train_loss < best_train_loss :
                tolerance = 0
                best_train_loss = train_loss
                best_model = model
        else:
                tolerance+=1
    if best_model is not None:
            model = best_model 
            test_metric =evaluate_unsupervised(model, problem, batch_size=args.batch_size,loss_fn=problem.loss_fn)
                
            print(test_metric)#, file=sys.stderr

    # sys.exit()
    sys.stdout.flush()
    #supervised learning
    tolerance = 0
    best_val_loss=100000
    best_val_acc=0
    best_result = None
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        # early stopping
        if tolerance > args.tolerance:
            break
        train_loss = 0
        
        X = problem.feats
        idx = np.random.permutation(X.shape[0])
        X_tilda = X[idx, :]
        
        n_mp = len(schemes)
        lbl_1 = torch.ones(n_mp, X.shape[0])
        lbl_2 = torch.zeros(n_mp, X.shape[0])
        lbl = torch.cat((lbl_1, lbl_2), 1)
        if torch.cuda.is_available():
            X_tilda = X_tilda.cuda()
            lbl = lbl.cuda()
        
        # Train unsupervised first
        _ = model.train()
        for ids, targets, epoch_progress in problem.iterate(mode='train', shuffle=True, batch_size=args.batch_size):

            optimizer.zero_grad()
            preds,weights,reg = model(X, X_tilda, None, None, None, get_embed=None)
            
            if weights is not None:
                weights=weights.cpu().detach().numpy()
                if len(weights.shape)>1 and weights.shape[0] != 1:
                    weights=np.sum(weights,axis=0)/weights.shape[0]
                # print(weights)

            loss = problem.loss_fn(preds[ids], targets.squeeze())
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            train_loss += loss.item()
            train_metric = problem.metric_fn(to_numpy(targets), to_numpy(preds[ids]))
        if verbose >=2:
            print(json.dumps({
                "epoch": epoch,
                "time": time() - start_time,
                "train_loss": train_loss,
                "train_metric": train_metric,
            }, double_precision=5))
            sys.stdout.flush()

        # Evaluate
        if epoch % args.log_interval == 0:
            _ = model.eval()
            loss, val_metric = evaluate(model, problem, batch_size=args.batch_size, mode='val',loss_fn=problem.loss_fn,coff=coff)
            _, test_metric =evaluate(model, problem, batch_size=args.batch_size, mode='test',loss_fn=problem.loss_fn,coff=coff)
            if val_metric['accuracy']>best_val_acc or (val_metric['accuracy']==best_val_acc and loss < best_val_loss):
                tolerance = 0
                best_val_loss = loss
                best_val_acc = val_metric['accuracy']
                best_result = json.dumps({
                "epoch": epoch,
                "test_metric": test_metric,
                "val_loss": loss,
                "val_metric": val_metric,
            }, double_precision=5)
            else:
                tolerance+=1
            
            if verbose >=2:
                print(json.dumps({
                 "epoch": epoch,
                 "val_loss": loss,
                 "val_metric": val_metric,
                 "test_metric": test_metric,
                 "tolerance:": tolerance,
                 }, double_precision=5))
                sys.stdout.flush()
    print('-- done --')
    print(best_result)
    print(best_result, file=sys.stderr)

    sys.stdout.flush()

    # if args.show_test:
    #     _ = model.eval()
    #     print(json.dumps({
    #         "test_metric": evaluate(model, problem, batch_size=args.batch_size, mode='test',loss_fn=problem.loss_fn,)
    #     }, double_precision=5))
