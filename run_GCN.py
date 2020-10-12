import subprocess
import ujson as json
import numpy as np

runs=10

args = [
    'python3',
    'baselines/runEmbedGCN.py',
    '--problem-path',
    './data/freebase/',
    '--problem',
    'yago',
    '--train-per',
    '0.4',
    '--hidden',
    '64',
    '--prep-dim',
    '128',
    '--dropout',
    '0.5',
    '--epochs',
    '2000'
]
print(args)
test_acc = []
test_macro = []
for seed in range(runs):
    process = subprocess.Popen(args+['--seed',str(seed)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    text = process.communicate()[1]

    lines = text.decode().split('\n')
    # print(lines)
    
    for line in lines:
        if '{' not in line:
            continue
        print(line)
        line = json.loads(line)
        if 'test_metric' in line:
            test_acc.append(line['test_metric']['accuracy'])
            test_macro.append(line['test_metric']['macro'])
test_acc = np.asarray(test_acc)
test_macro = np.asarray(test_macro)
print('average acc for {} runs is : {}'.format(len(test_acc), np.average(test_acc)))
print('average macro for {} runs is : {}'.format(len(test_macro), np.average(test_macro)))




