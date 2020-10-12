import subprocess
import ujson as json
import numpy as np

runs=10

args = [
    'python3',
    'baselines/runHAN.py',
    '--problem-path',
    './data/yelp/',
    '--problem',
    'yelp',
    '--train-per',
    '0.4',
    '--lr',
    '0.001',
    '--output-dims',
    '128,128,32,32',
    '--n-layer',
    '2',
    '--prep',
    '256',
    '--weight_decay',
    '5e-4',
    '--attn-dropout',
     '0.6',
     '--dropout',
     '0.4',
]
print(args)
test_acc = []
test_macro = []
for seed in range(runs):
    process = subprocess.Popen(args+['--seed',str(seed)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    text = process.communicate()[1]

    lines = text.decode().split('\n')
    print(lines)
    
    for line in lines:
        if '{' not in line:
            continue
        print(line)
        import sys
        sys.stdout.flush()
        line = json.loads(line)
        if 'test_metric' in line:
            test_acc.append(line['test_metric']['accuracy'])
            test_macro.append(line['test_metric']['macro'])
test_acc = np.asarray(test_acc)
test_macro = np.asarray(test_macro)
print('average acc for {} runs is : {}'.format(len(test_acc), np.average(test_acc)))
print('average macro for {} runs is : {}'.format(len(test_macro), np.average(test_macro)))




