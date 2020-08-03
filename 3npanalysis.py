from time import time
from time import strftime
import numpy as np
import sys
import os
import subprocess as sp
import pickle
from pathlib import Path
import itertools
import argparse
import multiprocessing as mp
import matplotlib.pyplot as plt
import random
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
ysnp = 0
gaps = np.linspace(0.4, 2, 10)
limits = np.linspace(0.6, 2, 10)
gaps2 = np.linspace(0.01, 0.15, 10)
limits2 = np.linspace(0.01, 0.1, 10)
'''
print(len(list(Path('./table_clutter/').glob('*.3np2table'))))
sets = []
count = {'{0:.3f} {1:.3f} {2:.3f} {3:.3f}'.format(g,l,g2,l2) : 0 for g,l,g2,l2 in itertools.product(gaps,limits,gaps2,limits2)}
for p in Path('./table_clutter/').glob('*.3np2table'):
    s1 = set()
    highest = 0
    for line in open(p,'r'):
        if line[0].isdigit():
            l = line.split()
            g,l,g2,l2,tq,tt = float(l[0]),float(l[1]),float(l[2]),float(l[3]),float(l[4]),float(l[5])
            count['{0:.3f} {1:.3f} {2:.3f} {3:.3f}'.format(g,l,g2,l2)] = (tt+tq)/2
            if (tt+tq)/2 > highest:
                highest = (tt+tq)/2
    best = {k: v for k, v in sorted(count.items(),reverse=True, key=lambda x: x[1]) if v>= 0.5 * highest}
    sets.append(set(best.keys()))

besties = dict()
for s in sets:
    for b in s:
        besties[b] = besties.get(b, 0) + 1
ranked = [(k,v) for k,v in sorted(besties.items(),reverse=True, key=lambda x: x[1])]
print(*ranked[:1000], sep='\n')

inter = set.intersection(*sets)
print(*inter, sep='\n')
print(len(inter))

# print(*zip(ranked,inter), sep='\n')
inter = ranked[:1000]
with open('params.pkl', 'wb') as p:
    pickle.dump(inter, p)
ysnp = 1
# '''
# scount = {k: v for k, v in sorted(count.items(),reverse=True, key=lambda x: x[1])[:150]}
# print(scount)

# print(len(sets), [len(s) for s in sets])
# intersection = sets[0] & sets[1] & sets[3] & sets[5] & sets[8] & sets[9]
# print(intersection)
# print(len(intersection))



    # sorted(d1, reverse=True, key=lambda x: float(x[2]))[:10], sep='/n')
if ysnp == 1:
    sys.exit()

files = ['/home/julius/biotech/sauzeror/listing_no-head.txt']
listing = dict()
for line in open(files[0],'r'):
    listing[line[:6]] = list(line[7:].rstrip().split('\t'))

qid = dict()
for i,line in enumerate(open('./dalilite/scope_140_targets.list', 'r')):
    lid = line.split()[2]
    qid[lid] = i
    qid[i] = lid

truth = dict()
for line in open('./dalilite/combinetable.pdb70', 'r'):
    l = line.split()
    truth[l[0]] = {i: s for i,s in enumerate(l[1])}
for pdb in random.sample(list(truth), 400):
    tr = [k for k,v in truth[pdb].items() if v != '-']
    if len(tr) > 0:
        print(qid[tr[0]], pdb)
        os.system('python 3new_params_sauzeror.py -mp align /home/julius/databases/dalilite_benchmark/structure_data/scope140_pdb/{0}.ent /home/julius/databases/dalilite_benchmark/structure_data/pdb_and_scope/{1}.pdb.gz -o ./table_clutter/{0}{1}.3np3table'.format(qid[tr[0]], pdb))

'''
for q in listing:
    t = random.choice(listing[q])
    # for t in ts:
    print(q,t)
    os.system('python 7new_params_sauzeror.py -mp align qu/c{0}.pdb da/c{1}.pdb -o {0}{1}.7np4table'.format(q,t))
'''
