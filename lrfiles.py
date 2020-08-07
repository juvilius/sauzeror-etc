import numpy as np
from scipy import spatial
from pathlib import Path
import multiprocessing as mp
import os, sys


def read_er(file):
    atoms = []
    er = []
    resnr=-200
    for line in open(file, 'r'):
        line = line.rstrip().split(',')
        if int(line[0]) < resnr:
            break
        resnr = int(line[0])
        atoms.append({'res_id': int(line[0]), 'res': line[1], 'coords': [float(s) for s in line[2:5]]})
        er.append(float(line[5]))
    return atoms, np.array(er)

def chunks(s, n):
    ''' yield successive n-sized chunks from s '''
    for i in range(0, len(s), n):
        yield s[i:i + n]

def scale2(x):
    x -= x.mean(axis=0)
    x /= np.sqrt(sum(x**2)/(x.shape[0] - 1))
    return x

def eigenrank(atom_coordinates):
    ''' calculating distance matrix -> adjacency matrices for different distance cutoffs -> PCA, scaling -> EigenRank '''
    if atom_coordinates.shape[1] != 3:
        atom_coordinates = atom_coordinates.T

    dm = spatial.distance_matrix(atom_coordinates, atom_coordinates, p=2)
    n = dm.shape[0]
    leaderranks = np.zeros((n,10))
    eg3 = np.ones(n+1)
    eg3[n] = 0  # ground node
    for a in range(5, 15):
        eg1 = np.ones((n+1,n+1))
        eg1[:n,:n] = np.greater_equal(a, dm)
        np.fill_diagonal(eg1, 0)
        h = 0
        eg1 = eg1.T/np.sum(eg1, axis=0)
        error = 10000
        error_threshold = 0.00002
        # eg2 = np.einsum('ij,j->i', eg1, eg3)
        eg2 = np.dot(eg1,eg3)
        for _ in range(3):
            # eg2 = np.einsum('ij,j->i', eg1, eg2)
            eg2 = np.dot(eg1,eg2)
        while error > error_threshold:
            M = eg2
            # eg2 = np.einsum('ij,j->i', eg1, eg2)
            eg2 = np.dot(eg1,eg2)
            error = np.sum(np.divide(np.abs(eg2-M),M))/(n+1)
            h+=1
        ground = eg2[n]/n
        leaderranks [:,a-5] = np.delete(eg2,-1)+ground
    return scale2(leaderranks)

class er_structure:
    def __init__(self, lines, id, file_id):
        self.atoms = []
        self.er = []
        for line in lines:
            line = line.rstrip().split(',')
            self.atoms.append({'res_id': int(line[0]), 'res': line[1], 'coords': [float(s) for s in line[2:5]]})
            self.er.append(float(line[5]))
        self.er = np.array(self.er)
        self.l = len(self.er)
        self.coordinates = np.asarray([a['coords'] for a in self.atoms])
        self.lr = eigenrank(self.coordinates)
        # with open('{}{}.lr'.format('structures/', self.id), 'wt') as f:
        f.write('# {}\n'.format(id))
        for atoms, er, lr in zip(self.atoms, self.er, [k for k in self.lr]):
            f.write('{:05d},{},{:.3f},{:.3f},{:.3f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(atoms['res_id'], atoms['res'].lower(), *atoms['coords'], er, *lr))
        f.write('\n')

# input1 = [line.rstrip().split()[2] for line in open('scope_140_targets.list')]
# input2 = [p for p in Path('scope140-structures/').glob('*.er')]
# for i in input1:
#     if len([p for p in input2 if i in str(p)]) < 1:
#         print(i)
f = 0
input2 = [p for p in Path('structures/').glob('*.er')]
for e in input2:
    file_id = os.path.basename(e).split('.')[0]
    f = open('structures/{}.lr'.format(file_id), 'w')
    for line in open(e, 'r'):
        if line.startswith('#'):
            lines = []
            id = line.rstrip().split()[1]
        elif not line.startswith('\n'):
            lines.append(line)
        else:
            er_structure(lines, id, file_id)
    f.close()



sys.exit()

'''
input2 = [p for p in Path('structures/').glob('*.lr')]
for i, chunk in enumerate(chunks(input2, 2**13)):
    with open('strctrs/{:02d}.lr'.format(i), 'w') as f:
        for p in chunk:
            f.write('# {}\n'.format(os.path.basename(p).split('.')[0]))
            [f.write(line) for line in open(p, 'r')]
            f.write('\n')
            os.remove(p)
input2 = [p for p in Path('structures/').glob('*.er')]
for i, chunk in enumerate(chunks(input2, 2**13)):
    with open('strctrs/{:02d}.er'.format(i), 'w') as f:
        for p in chunk:
            f.write('# {}\n'.format(os.path.basename(p).split('.')[0]))
            [f.write(line) for line in open(p, 'r')]
            f.write('\n')
            os.remove(p)
sys.exit()
# '''

f = open('scope140.lr', 'w')
input2 = [p for p in Path('scope140-structures/').glob('*.er')]
[er_structure(i) for i in input2]
f.close()
