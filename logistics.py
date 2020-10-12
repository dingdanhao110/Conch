
from __future__ import division
from __future__ import print_function

from models import LogisticRegressionModel
from functools import partial
import sys
import argparse
import ujson as json
import numpy as np
from time import time
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from problem import NodeProblem
from helpers import set_seeds, to_numpy
from lr import LRSchedule

def set_progress(optimizer, lr_scheduler, progress):
    lr = lr_scheduler(progress)
    LRSchedule.set_lr(optimizer, lr)

def train_step(model, optimizer, x, targets, loss_fn):
    optimizer.zero_grad()
    preds = model(x)
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
        pred = model(features[ids])
        loss += loss_fn(pred, targets.squeeze()).item()
        preds.append(to_numpy(pred))
        acts.append(to_numpy(targets))
    #
    return loss, problem.metric_fn(np.vstack(acts), np.vstack(preds))

def read_embed(n_target, path="./data/freebase/",
               emb_file="MADW_16", mp2vec=False):
    if mp2vec:
        embedding = []
        with open("{}{}.emb".format(path, emb_file)) as f:
            n_nodes, n_feature = map(int, f.readline().strip().split())
            n_nodes-=1
            for line in f:
                arr = line.strip().split()
                if str(arr[0])[0] == '<': continue
            # embedding.append(list(map(float,arr[1:])))
                embedding.append([ int(str(arr[0])[1:]) ]+ list(map(float,arr[1:])))
            embedding = np.asarray(embedding)
    else:
        with open("{}{}.emb".format(path, emb_file)) as f:
            n_nodes, n_feature = map(int, f.readline().strip().split())
        embedding = np.loadtxt("{}{}.emb".format(path, emb_file),
                           dtype=np.float32, skiprows=1,encoding='latin-1')
    
    print("number of nodes:{}, embedding size:{}".format(n_nodes, n_feature))
    
    emb_index = {}
    for i in range(n_nodes):
        # if type(embedding[i, 0]) is not int:
        #     continue
        emb_index[embedding[i, 0]] = i

    features = np.asarray([embedding[emb_index[i], 1:] for i in range(n_target)])

    # assert features.shape[1] == n_feature
    # assert features.shape[0] == n_nodes

    return features, n_target, n_feature


# # --
# Args

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--problem-path', type=str, required=True)
    parser.add_argument('--problem', type=str, required=True)
    parser.add_argument('--feat', type=str, required=True)
    parser.add_argument('--no-cuda', action="store_true")
    parser.add_argument('--mp2vec', action="store_true")
    # Optimization params
    parser.add_argument('--batch-size', type=int, default=999999)
    parser.add_argument('--epochs', type=int, default=0)
    parser.add_argument('--lr-init', type=float, default=0.01)
    parser.add_argument('--lr-schedule', type=str, default='constant')
    parser.add_argument('--tolerance', type=int, default=10)
    parser.add_argument('--weight-decay', type=float, default=0.0)

    parser.add_argument('--n-train-samples', type=str, default='8,8')
    parser.add_argument('--n-val-samples', type=str, default='8,8')
    parser.add_argument('--output-dims', type=str, default='16,16')
    parser.add_argument('--train-per', type=float, default=0.4)
    # Logging
    parser.add_argument('--log-interval', default=1, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--show-test', action="store_true")

    # --
    # Validate args

    args = parser.parse_args()
    args.cuda = not args.no_cuda
    assert args.batch_size > 1, 'parse_args: batch_size must be > 1'
    return args


if __name__ == "__main__":
    args = parse_args()
    set_seeds(args.seed)

    # --
    # Load problem
    s={'dblp':['APA'],'yelp': ['BRURB'], 'yago': ['MAM']}
    schemes = s[args.problem]
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    problem = NodeProblem(problem_path=args.problem_path, problem=args.problem, device=device, schemes=schemes, input_edge_dims = 18,train_per=args.train_per)

    n_targets = problem.targets.shape[0]
    #load embeddings as features
    features, n_nodes, n_feature = read_embed(n_targets,
                                              path=args.problem_path,
                                              emb_file=args.feat,
                                              mp2vec=args.mp2vec)
    features = torch.FloatTensor(features)
    # --
    # Define model

    n_train_samples = list(map(int, args.n_train_samples.split(',')))
    n_val_samples = list(map(int, args.n_val_samples.split(',')))
    output_dims = list(map(int, args.output_dims.split(',')))
    model = LogisticRegressionModel(**{
        "input_dim": n_feature,
        "output_dim": problem.n_classes,
    })

    if args.cuda:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
        model = model.to(device)
        features.to(device)

    # --
    # Define optimizer

    lr_scheduler = partial(getattr(LRSchedule, args.lr_schedule), lr_init=args.lr_init)
    lr = lr_scheduler(0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    # print(model, file=sys.stdout)

    # --
    # Train

    set_seeds(args.seed)

    # start_time = time()
    # val_metric = None
    # tolerance = 0
    # best_val_loss=100000
    # best_result = None
    # for epoch in range(args.epochs):
    #     # early stopping
    #     if tolerance > args.tolerance:
    #         break
    #     train_loss = 0
    #     # Train
    #     _ = model.train()
    #     for ids, targets, epoch_progress in problem.iterate(mode='train', shuffle=True, batch_size=args.batch_size):
    #         set_progress(optimizer, lr_scheduler, (epoch + epoch_progress) / args.epochs)
    #         loss, preds = train_step(
    #             model=model,
    #             optimizer=optimizer,
    #             x=features[ids],
    #             targets=targets,
    #             loss_fn=problem.loss_fn,
    #         )
    #         train_loss += loss.item()
    #         train_metric = problem.metric_fn(to_numpy(targets), to_numpy(preds))
    #         # print(json.dumps({
    #         #     "epoch": epoch,
    #         #     "epoch_progress": epoch_progress,
    #         #     "train_metric": train_metric,
    #         #     "time": time() - start_time,
    #         # }, double_precision=5))
    #         # sys.stdout.flush()

    #     # print(json.dumps({
    #     #     "epoch": epoch,
    #     #     "time": time() - start_time,
    #     #     "train_loss": train_loss,
    #     # }, double_precision=5))
    #     # sys.stdout.flush()

    #     # Evaluate
    #     if epoch % args.log_interval == 0:
    #         _ = model.eval()
    #         loss, val_metric = evaluate(model, problem, batch_size=args.batch_size, mode='val',loss_fn=problem.loss_fn,)
    #         _, test_metric =evaluate(model, problem, batch_size=args.batch_size, mode='test',loss_fn=problem.loss_fn,)
    #         print(json.dumps({
    #             "epoch": epoch,
    #             "val_loss": loss,
    #             "val_metric": val_metric,
    #             "test_metric": test_metric,
    #             "tolerance:": tolerance,
    #         }, double_precision=5))
    #         sys.stdout.flush()

    #         if loss < best_val_loss:
    #             tolerance = 0
    #             best_val_loss = loss
    #             best_result = json.dumps({
    #             "epoch": epoch,
    #             "val_loss": loss,
    #             "val_metric": val_metric,
    #             "test_metric": test_metric,
    #         }, double_precision=5)
    #         else:
    #             tolerance+=1

    # print('-- done --', file=sys.stderr)
    # print(best_result)
    # sys.stdout.flush()

    # print('-- sklearn --', file=sys.stderr)

    features = features.numpy()
    for ids, targets, epoch_progress in problem.iterate(mode='train', shuffle=True, batch_size=999999):
        x_train = features[ids]
        y_train = targets
    for ids, targets, epoch_progress in problem.iterate(mode='val', shuffle=True, batch_size=999999):
        x_val = features[ids]
        y_val = targets
    for ids, targets, epoch_progress in problem.iterate(mode='test', shuffle=True, batch_size=999999):
        x_test = features[ids]
        y_test = targets

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    # fpr, tpr, thresholds = metrics.roc_curve(y_test, y_scores, pos_label=2)
    # plot_roc_curve(fpr,tpr,'ROC')
    y_pred=logreg.predict(x_test)
    y_train_scores = logreg.predict_proba(x_train)[:, 1]
    y_test_scores = logreg.predict_proba(x_test)[:, 1]
    print('Accuracy of Logistic regression classifier on training set: {:.4f}'
          .format(logreg.score(x_train, y_train)))
    print('Accuracy of Logistic regression classifier on test set: {:.4f}'
          .format(logreg.score(x_test, y_test)))
    print('Macro of Logistic regression classifier on test set: {:.4f}'
          .format(f1_score(y_test,y_pred, average='macro')))
    print('Micro of Logistic regression classifier on test set: {:.4f}'
          .format(f1_score(y_test,y_pred, average='micro')))

    result = json.dumps({
                "test_metric": {'accuracy':f1_score(y_test,y_pred, average='micro'),
                'macro':f1_score(y_test,y_pred, average='macro')
                },
            }, double_precision=5)
    print(result,file=sys.stderr)
    # from sklearn.neighbors import KNeighborsClassifier

    # knn = KNeighborsClassifier()
    # knn.fit(x_train, y_train)
    # y_train_scores = knn.predict_proba(x_train)[:, 1]
    # y_test_scores = knn.predict_proba(x_test)[:, 1]
    # print('Accuracy of K-NN classifier on training set: {:.4f}'
    #       .format(knn.score(x_train, y_train)))
    # print('Accuracy of K-NN classifier on test set: {:.4f}'
    #       .format(knn.score(x_test, y_test)))

    # from sklearn.svm import SVC

    # svm = SVC(probability=True)
    # svm.fit(x_train, y_train)
    # y_train_scores = svm.predict_proba(x_train)[:, 1]
    # y_test_scores = svm.predict_proba(x_test)[:, 1]
    # print('Accuracy of SVM classifier on training set: {:.4f}'
    #       .format(svm.score(x_train, y_train)))
    # print('Accuracy of SVM classifier on test set: {:.4f}'
    #       .format(svm.score(x_test, y_test)))




