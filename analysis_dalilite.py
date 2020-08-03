from time import time
from time import strftime
import numpy as np
import sys
import os
import pickle
from pathlib import Path
import itertools
import argparse
import multiprocessing as mp
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

scope_cla = '../dir.cla.scope.2.07-stable.txt'

def read_qrf(file):
    ''' read query result file, return id and ranked pdb id's '''
    id = os.path.basename(file)[:7]
    all = []
    for line in open(file):
        if not (line.startswith('#') or line.startswith('\n')):
            all.append(line.split())

    return id,[i[0] for i in sorted(all, reverse=True, key=lambda x: float(x[-1]))]


tests = [
# '6-sauzeror-2.2-4.2.txt',
# 'LR-refinement-9-sauzeror-2.2-4.2-0.4-0.2.txt',
# 'premier-sauzeror.txt',
'refinement-3-sauzeror-0.8-0.7-0.4-0.2.txt',
# 'refinement-3-sauzeror.txt'
]
scoring = 'tm1'

# 0     1       2       3       4       5       6       7       8       9       10
# chain	l	lali	gaps	score	rmsd	gdt_ts	gdt_sim	tm1	tm2	tm

for t in tests:
    # for 'evaluate_ordered_lists.pl':
    querywise_raw = './ordered_querywise/{}_{}/'.format(scoring, t[:-4])
    Path(querywise_raw).mkdir(exist_ok=True)
    def ordered_list(file):
        id = os.path.basename(file)[:7]
        all = []
        for line in open(file):
            if not (line.startswith('#') or line.startswith('\n')):
                line = line.split()
                # if line[0] in pdb70:
                all.append(line)
        ordered = [(i[0],id,i[7]) for i in sorted(all, reverse=True, key=lambda x: float(x[8]))]
        with open(querywise_raw+id, 'w') as f:
            for l in ordered:
                f.write('{}\t{}\t{}\n'.format(*l))

    results_name = scoring+'_'+t[:-4]+'_pdb'
    for r in Path('results/').glob('*'+t):
        ordered_list(r)
    # os.system('./bin/evaluate_ordered_lists.pl {0} combinetable.pdb70 scope_140_targets.list querywise > evaluation_results/{1}'.format(querywise_raw, results_name))
    ### FULL PDB
    os.system('./bin/evaluate_ordered_lists.pl {0} combinetable.pdb scope_140_targets.list querywise > evaluation_results/{1}'.format(querywise_raw, results_name))
    os.system('./bin/evaluate_ordered_lists.pl {0} combinetable.pdb70 scope_140_targets.list querywise > evaluation_results/{1}'.format(querywise_raw, results_name+'70'))


sys.exit()


qid = dict()
for i,line in enumerate(open('scope_140_targets.list', 'r')):
    lid = line.split()[2]
    qid[lid] = i
    qid[i] = lid

def ture(): # truth dictionaries
    ''' levels:  1    fold level match
                 2    superfamily level match
                 3    family level match '''
    sccs = dict()
    for line in open(scope_cla, 'r'):
        if not line.startswith('#'):
            l = line.split()
            cid = '.'.join(l[3].split('.')[:2])
            sccs[l[0]] = cid

    truth = dict()
    foldclasses = dict()
    for line in open('combinetable.pdb70', 'r'):
        l = line.split()
        sid = l[0] # SCOPe ID
        cids = l[2:]
        for c in cids:
            cid = '.'.join(c.split('.')[:2])
            foldclasses[cid] = foldclasses.get(cid, 0) + 1
        truth[sid] = {i: s for i,s in enumerate(l[1])}

    return sccs,truth,foldclasses

def fmax(id,rank,truths,level):
    ''' take id and ranked pdb, output fmax '''
    sccs,truth,foldclasses = truths
    f = []
    total = foldclasses[sccs[id]]
    tp = 0
    for n,p in enumerate(rank):
        if truth[p][qid[id]] == str(level):
            tp +=1
        f.append((2*tp)/(n+total))
    # plt.plot(f);plt.show()
    return max(f)

def dalli():
    fmaxs = []
    for level in [1,2,3]:
        truths = ture(level)
        for r in Path('.').glob('*dalilite_benchmark_pdb70.txt'):
            id, rank = read_qrf(r)
            fm = fmax(id,rank,truths,level)
            fmaxs.append(fm)
        print(level, np.mean(fmaxs))
# dalli()
# print({k:v for k,v in ture()[2].items() if v == 0})
print(len(ture()[2]))
