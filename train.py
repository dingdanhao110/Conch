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
from problem import NodeProblem
from helpers import set_seeds, to_numpy
from nn_modules import aggregator_lookup, prep_lookup, sampler_lookup, edge_aggregator_lookup, \
    metapath_aggregator_lookup
from lr import lr_schedule
from bipartite import BipartiteGCN
from cling import CLING

# --
# Helpers

def set_progress(optimizer, lr_scheduler, progress):
    lr = lr_scheduler(progress)
    LRSchedule.set_lr(optimizer, lr)


def train_step(model, optimizer, ids, targets, loss_fn):
    optimizer.zero_grad()
    preds,weights = model(ids, train=True)
    if weights is not None:
        weights=weights.cpu().detach().numpy()
        if len(weights.shape)>1 and weights.shape[0] is not 1:
            weights=np.sum(weights,axis=0)/weights.shape[0]
        print(weights)
    loss = loss_fn(preds, targets.squeeze())
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    return loss, preds


def evaluate(model, problem, batch_size, loss_fn, mode='val'):
    assert mode in ['test', 'val']
    preds, acts = [], []
    loss=0
    for (ids, targets, _) in problem.iterate(mode=mode, shuffle=False, batch_size=batch_size):
        # print(ids.shape,targets.shape)
        pred, _= model(ids, train=False)
        loss += loss_fn(pred, targets.squeeze()).item()
        preds.append(to_numpy(pred))
        acts.append(to_numpy(targets))
    #
    return loss, problem.metric_fn(np.vstack(acts), np.vstack(preds))


# def evaluate(model, problem, batch_size, mode='val'):
#     assert mode in ['test', 'val']
#     preds, acts = [], []
#     for (ids, targets, _) in problem.iterate(mode=mode, shuffle=False, batch_size=batch_size):
#         # print(ids.shape,targets.shape)
#         preds.append(to_numpy(model(ids, train=False)))
#         acts.append(to_numpy(targets))
#
#     return problem.metric_fn(np.vstack(acts), np.vstack(preds))
# # --
# Args

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--problem-path', type=str, default='data/dblp2/')
    parser.add_argument('--problem', type=str, default='dblp')

    parser.add_argument('--no-cuda', action="store_true",default=False)

    # Optimization params
    parser.add_argument('--batch-size', type=int, default=99999)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--lr-init', type=float, default=0.001)
    parser.add_argument('--lr-schedule', type=str, default='constant')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batchnorm', action="store_true")
    parser.add_argument('--tolerance', type=int, default=100)
    parser.add_argument('--attn-dropout',type=float,default=0)
    # Architecture params
    parser.add_argument('--sampler-class', type=str, default='sparse_uniform_neighbor_sampler')

    parser.add_argument('--prep-class', type=str, default='linear')  # linear,identity
    parser.add_argument('--prep-len', type=int, default=128)
    parser.add_argument('--in-edge-len', type=int, default=128)
    parser.add_argument('--aggregator-class', type=str, default='sum')
    parser.add_argument('--edge-aggr-class', type=str, default='sum')
    parser.add_argument('--mpaggr-class', type=str, default='attention')

    parser.add_argument('--concat-node', action="store_true",default=False)
    parser.add_argument('--concat-edge', action="store_true")

    parser.add_argument('--n-head', type=int, default=4)
    parser.add_argument('--k', type=int, default=10)
    # parser.add_argument('--n-train-samples', type=str, default='600,600')
    # parser.add_argument('--n-val-samples', type=str, default='600,600')
    parser.add_argument('--output-dims', type=str, default='64,32,32,32')
    parser.add_argument('--n-layer', type=int, default='2')
    
    parser.add_argument('--train-per', type=float, default=0.4)

    # Logging
    parser.add_argument('--log-interval', default=1, type=int)
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
    torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.deterministic = True
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

    # --
    # Define model

    # n_train_samples = list(map(int, args.n_train_samples.split(',')))
    # n_val_samples = list(map(int, args.n_val_samples.split(',')))
    output_dims = list(map(int, args.output_dims.split(',')))
    model = BipartiteGCN(**{
        "problem": problem,
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
        # model = model.half()
        model = model.to(device)

    # --
    # Define optimizer
    lr_scheduler = partial(getattr(LRSchedule, args.lr_schedule), lr_init=args.lr_init)
    lr = lr_scheduler(0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=args.weight_decay,momentum=0.9)
    print(model, file=sys.stdout)
    if args.cuda:
        print('GPU memory allocated: ', torch.cuda.memory_allocated() / 1000 / 1000 / 1000)
    # --
    # Train

    set_seeds(args.seed)

    start_time = time()
    val_metric = None
    tolerance = 0
    best_val_loss=100000
    best_val_acc=0
    best_result = None
    
    if args.lr_schedule=='cosine':
        Ti=1
        mult=2
        Tcur=0

    for epoch in range(args.epochs):
        # early stopping
        if tolerance > args.tolerance:
            break
        train_loss = 0
        
        # Train
        _ = model.train()
        for ids, targets, epoch_progress in problem.iterate(mode='train', shuffle=True, batch_size=args.batch_size):
            if args.lr_schedule=='cosine':
                lr = lr_scheduler(Tcur + epoch_progress, epochs=Ti)
                LRSchedule.set_lr(optimizer, lr)
                print('learning rate:{}'.format(lr))
            else:
                # set_progress(optimizer, lr_scheduler, (epoch + epoch_progress) / args.epochs)
                pass
            loss, preds = train_step(
                model=model,
                optimizer=optimizer,
                ids=ids,
                targets=targets,
                loss_fn=problem.loss_fn,
            )
            train_loss += loss.item()
            train_metric = problem.metric_fn(to_numpy(targets), to_numpy(preds))
            #print(json.dumps({
            #    "epoch": epoch,
            #    "epoch_progress": epoch_progress,
            #    "train_metric": train_metric,
            #    "time": time() - start_time,
            #}, double_precision=5))
            #sys.stdout.flush()

        print(json.dumps({
            "epoch": epoch,
            "time": time() - start_time,
            "train_loss": train_loss,
        }, double_precision=5))
        sys.stdout.flush()

        #update learning rate for cosine annealing
        if args.lr_schedule=='cosine':
            if Tcur%Ti==0 and Tcur>0:
                Ti*=mult
                Tcur=0
            else:
                Tcur+=1

        # Evaluate
        if epoch % args.log_interval == 0:
            _ = model.eval()
            loss, val_metric = evaluate(model, problem, batch_size=args.batch_size, mode='val',loss_fn=problem.loss_fn,)
            _, test_metric =evaluate(model, problem, batch_size=args.batch_size, mode='test',loss_fn=problem.loss_fn,)
            if val_metric['accuracy']>best_val_acc or (val_metric['accuracy']==best_val_acc and loss < best_val_loss):
                tolerance = 0
                best_val_loss = loss
                best_val_acc = val_metric['accuracy']
                best_result = json.dumps({
                "epoch": epoch,
                "val_loss": loss,
                "val_metric": val_metric,
                "test_metric": test_metric,
            }, double_precision=5)
            else:
                tolerance+=1
            
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
