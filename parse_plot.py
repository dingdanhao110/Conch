import sys
import argparse
import ujson as json
import matplotlib.pyplot as plt

from time import time


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--log-path', type=str, default='plot')
    parser.add_argument('--log-file', type=str, default='')

    args = parser.parse_args()

    return args


def plot_loss(args):
    epoches = []
    val_loss = []
    train_loss = []
    val_acc = []
    train_acc = []
    test_acc = []

    with open('{}{}.txt'.format(args.log_path, args.log_file), mode='r') as f:
        print('file: {}{}.txt'.format(args.log_path, args.log_file))
        for line in f:
            if '{' not in line:
                continue

            line = json.loads(line)
            if 'train_loss' in line:
                train_loss.append(line['train_loss'])
            if 'val_loss' in line:
                epoches.append(line['epoch'])
                val_loss.append(line['val_loss'])

            # if 'train_metric' in line:
            #     if line['epoch_progress'] == 0:
            #         train_acc.append(line['train_metric']['accuracy'])
            if 'val_metric' in line:
                val_acc.append(line['val_metric']['accuracy'])
            if 'test_metric' in line:
                test_acc.append(line['test_metric']['accuracy'])

    l = min(len(train_loss), len(val_loss), len(val_acc), len(test_acc), len(epoches))

    train_loss = train_loss[:l]
    val_loss = val_loss[:l]
    test_acc = test_acc[:l]
    val_acc = val_acc[:l]
    epoches = epoches[:l]

    plt.figure()
    plt.title('file: {}{}.txt'.format(args.log_path, args.log_file))
    plt.subplot(211)
    plt.plot(epoches, train_loss, 'r', label='train')
    plt.plot(epoches, val_loss, 'g', label='val')
    plt.ylabel('loss')
    plt.xlabel('epoches')
    plt.legend(loc='upper right')

    plt.subplot(212)
    plt.plot(epoches, test_acc, 'r', label='test')
    plt.plot(epoches, val_acc, 'g', label='val')
    plt.ylabel('accuracy')
    plt.xlabel('epoches')
    plt.legend(loc='lower right')
    plt.suptitle('file: {}{}.txt'.format(args.log_path, args.log_file))
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    plot_loss(args)
    #
    # print(args.log_path)
    # best = None
    # best_acc = 0
    # best_config = None
    # test_acc = -1
    # same_best = 0
    #
    # lr = ["0.001", "0.0005", "0.0003", "0.0001," "0.00005"]
    # prep = ["linear"]
    # aggr = ["attention"]
    # head = ["1", "2", "4", "8"]
    # dropout = ["0", "0.1", "0.3", "0.5", "0.7"]
    # attn_dropout = ["0", "0.1", "0.3", "0.5", "0.7"]
    # count = 1
    # for layer in range(3):
    #     for l in lr:
    #         for p in prep:
    #             for a in aggr:
    #                 for h in head:
    #                     for d in dropout:
    #                         for ad in attn_dropout:
    #                             try:
    #                                 with open('{}_{}.txt'.format(args.log_path, count), mode='r') as f:
    #                                     # print('success')
    #                                     for line in f:
    #                                         if '{' not in line:
    #                                             continue
    #
    #                                         line = json.loads(line)
    #
    #                                         if 'test_metric' in line:
    #                                             test_acc = line['test_metric']['accuracy']
    #                                             test_metric = line['test_metric']
    #                             except OSError:
    #                                 pass
    #                             if test_acc == best_acc:
    #                                 same_best += 1
    #                             if test_acc > best_acc:
    #                                 same_best = 1
    #                                 best_acc = test_acc
    #                                 best = test_metric
    #                                 best_config = json.dumps({
    #                                     "lr": l,
    #                                     "prep": p,
    #                                     "aggr": a,
    #                                     "head": h,
    #                                     "dropout": d,
    #                                     "attn_dropout": ad,
    #                                     "n_layer": count // 500 + 1,
    #                                     "count": count,
    #                                 })
    #
    #                             count += 1
    #                             pass
    #
    # print('best configuration: ', best_config)
    # print('best result: ', best)
    # print('number of same results: ', same_best)
    # args.log_file = '_{}'.format(json.loads(best_config)['count'])
    # plot_loss(args)
    # pass
