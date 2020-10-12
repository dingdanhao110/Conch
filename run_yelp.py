import subprocess
import ujson as json
import numpy as np

runs=10
#Top k HAN, variant2； adjust train_per in helper.py
args = [
    'python3',
    'train_reg4.py',
   '--problem-path',
    'data/yelp/',
    '--problem',
    'yelp',
    '--lr-init',
    '1e-3',
    '--weight-decay',
    '5e-4',
    '--dropout',
    '0.5',
    '--prep-class',
    'linear',
    '--prep-len',
    '256',
    '--in-edge-len',
    '16',
    '--n-head',
    '1',
    '--k',
    '10',
    '--output-dims',
    '128,64,32,32',
    '--n-layer',
    '1',
    # '--max-coff',
    # '1',
    # '--coff-exp',
    # '20',
    '--train-per',
    '0.2',
]
print(args)
test_acc = []
test_macro = []
val_acc = []
val_macro = []
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
            val_acc.append(line['val_metric']['accuracy'])
            val_macro.append(line['val_metric']['macro'])
test_acc = np.asarray(test_acc)
test_macro = np.asarray(test_macro)
val_acc = np.asarray(val_acc)
val_macro = np.asarray(val_macro)
print('average acc for {} runs is : {}'.format(len(test_acc), np.average(test_acc)))
print('average macro for {} runs is : {}'.format(len(test_macro), np.average(test_macro)))
print('average val acc for {} runs is : {}'.format(len(val_acc), np.average(val_acc)))
print('average val macro for {} runs is : {}'.format(len(val_macro), np.average(val_macro)))



