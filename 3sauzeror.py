#╔════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
#║  .oooooo..o       .o.       ooooo     ooo  oooooooooooo oooooooooooo ooooooooo.     .oooooo.   ooooooooo.  ║
#║ d8P'    `Y8      .888.      `888'     `8' d'""""""d888' `888'     `8 `888   `Y88.  d8P'  `Y8b  `888   `Y88.║
#║ Y88bo.          .8"888.      888       8        .888P    888          888   .d88' 888      888  888   .d88'║
#║  `"Y8888o.     .8' `888.     888       8       d888'     888oooo8     888ooo88P'  888      888  888ooo88P' ║
#║      `"Y88b   .88ooo8888.    888       8     .888P       888    "     888`88b.    888      888  888`88b.   ║
#║ oo     .d8P  .8'     `888.   `88.    .8'    d888'    .P  888       o  888  `88b.  `88b    d88'  888  `88b. ║
#║ 8""88888P'  o88o     o8888o    `YbodP'    .8888888888P  o888ooooood8 o888o  o888o  `Y8bood8P'  o888o  o888o║
#╚══════════════╦══════╦════════════════════════════════════════════════════════════╦══════╦══════════════════╝
#               ║      ║ Structure Alignments Using Z-scaled EigenRanks Of Residues ║      ║
#               ║      ╚════════════════════════════════════════════════════════════╝      ║
#               ║  ┌────────────────────────────────────────────────────────────────────┐  ║
#               ╟──┤ python sauzeror.py [1..] align input1 input2 [2..] [-o result.txt] │  ║
#               ║  └────────────────────────────────────────────────────────────────────┘  ║
#               ║  ┌──────────────────────────────────────────────────────┐                ║
#               ╟──┤ an input can be                                      │                ║
#               ║  │    (1) a file name,                                  │                ║
#               ║  │    (2) a directory (filtering for .pdb, .ent, .atm), │                ║
#               ║  │    (3) a file with a list of files; one per line.    │                ║
#               ║  │        (e.g. find ~/path/to/pdb -name '*.pdb')       │                ║
#               ║  │ these 3 options can be mixed.                        │                ║
#               ║  │                                                      │                ║
#               ║  │ alignment parameters may be given at the end ([2..]) │                ║
#               ║  │   --gap-cost     gap cost parameter (default 0.5)    │                ║
#               ║  │   --limit        limit parameter (default 1.4)       │                ║
#               ║  │                                                      │                ║
#               ╟──┤   -o  --output     output file                       │                ║
#               ║  └──────────────────────────────────────────────────────┘                ║
#               ║  ┌───────────────────────────────────────────────────────────────────┐   ║
#               ╟──┤ further options may be given instead of [1..]                     │   ║
#               ║  │   -h   --help              show this help message                 │   ║
#               ║  │   -v   --verbose           for more verbosity (starting with #)   │   ║
#               ║  │   -mp  --multiprocessing   use multiprocessing                    │   ║
#               ║  │   -nc  --no-cache          don't cache numbas machine code        │   ║
#               ╟──┤   -nb  --no-banner         don't use the the banner in the output │   ║
#               ║  └───────────────────────────────────────────────────────────────────┘   ║
#               ╚══════════════════════════════════════════════════════════════════════════╝
from time import time
from time import strftime
import numpy as np
from numba import jit, float32, int32, types
from scipy import spatial
from scipy import stats
from scipy.special import ndtr
import sys,os,pickle,re
import tarfile, io
from pathlib import Path
import itertools
import argparse
import multiprocessing as mp
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# term_width = os.get_terminal_size().columns
logo = []
for i,line in enumerate(open(__file__, 'r')):
    if 1 <= i < 8:
        logo.append((line[2:-2]))
helptext = []
for i,line in enumerate(open(__file__, 'r')):
    if line.startswith('#'):
        helptext.append((line[1:].rstrip()))
    else:
        break

### ARGPARSE

parser = argparse.ArgumentParser('SAUZEROR', add_help=False)
subparser = parser.add_subparsers()

parser_a = subparser.add_parser('align', help='align sets of protein structures pairwise')
parser_a.add_argument('input1', type=str, help='first input')
parser_a.add_argument('input2', type=str, help='second input')
parser_a.add_argument('--gap-cost', type=float, default=0.7, help='gap cost')
parser_a.add_argument('--limit', type=float, default=0.8, help='limit parameter for Smith-Waterman-algorithm')
parser_a.add_argument('--gap2', type=float, default=0.4, help='gap cost')
parser_a.add_argument('--limit2', type=float, default=0.2, help='limit parameter for Smith-Waterman-algorithm')
parser_a.add_argument('-o', '--output', default=False, help='output to file instead of stdout')

# future feature
parser_c = subparser.add_parser('classify', help='classify a set of proteins with SCOPe')
parser_c.add_argument('input', type=str, help='input structure')

parser.add_argument('-v', '--verbose', default=False, action='store_true', help='verbose output')
parser.add_argument('-nb', '--no-banner', default=False, action='store_true', help='no logo in output')
parser.add_argument('-mp', '--multiprocessing', default=False, action='store_true')
parser.add_argument('-nc', '--no-cache', action='store_false', default=True, help='DON\'T write numba machine code to cache')

parser.add_argument('-h', '--help', action='store_true', help='that\'s better')
args = parser.parse_args()
if args.help:
    print('\n'.join(helptext))
if hasattr(args,'input'):
    print('classification of ', args.input)
    sys.exit()
elif hasattr(args,'input1'):
    if os.path.isdir(args.input1):
        input1 = list(Path(args.input1).rglob('*.pdb'))
        input1.extend(Path(args.input1).rglob('*.ent'))
        input1.extend(Path(args.input1).rglob('*.atm'))
    elif os.path.isfile(args.input1):
        input1 = []
        for line in open(args.input1,'r'):
            line=line.rstrip()
            if not line.startswith('#'):
                if not os.path.isfile(line):
                    input1 = [os.path.abspath(args.input1)]
                    break
                else:
                    input1.append(os.path.abspath(line.rstrip()))
    if os.path.isdir(args.input2):
        input2 = list(Path(args.input2).rglob('*.pdb'))
        input2.extend(Path(args.input2).rglob('*.ent'))
        input2.extend(Path(args.input2).rglob('*.atm'))
    elif os.path.isfile(args.input2):
        input2 = []
        for line in open(args.input2,'r'):
            line=line.rstrip()
            if not line.startswith('#'):
                if not os.path.isfile(line):
                    input2 = [os.path.abspath(args.input2)]
                    break
                else:
                    input2.append(os.path.abspath(line.rstrip()))
else:
    if not args.help:
        print('\n'.join(helptext))
    sys.exit()

### GENERAL FUNCTIONS

def dm_euclidian(a,b):
    ''' euclidian distance matrix of 2 sequences '''
    # a,b = a[:,0],b[:,0]
    m,n = len(a),len(b)
    result = np.empty((m,n),dtype=np.single)
    if m < n:
        for i in range(m):
            result[i,:] = np.abs(a[i]-b)
    else:
        for j in range(n):
            result[:,j] = np.abs(a-b[j])
    return result

def dm_normd(a,b):
    ''' distance matrix for 2 normally distributed sequences '''
    m,n = len(a),len(b)
    result = np.zeros((m,n), dtype=np.single)
    if m < n:
        for i in range(m):
            result[i,:] = np.abs(ndtr(a[i])-ndtr(b))
    else:
        for j in range(n):
            result[:,j] = np.abs(ndtr(a)-ndtr(b[j]))
    return result

def rmsd(a,b):
    ''' a and b are vectors '''
    diff = np.array(a) - np.array(b)
    n = len(a)
    return np.sqrt((diff * diff).sum() / n)
    # return np.sqrt(((a-b)**2).sum() / n)

def chunks(s, n):
    ''' yield successive n-sized chunks from s '''
    for i in range(0, len(s), n):
        yield s[i:i + n]

###################################################
# toggle for residue one-letter and 3-letter code #
###################################################

lettercode = '''Ala,A,Alanine
Arg,R,Arginine
Asn,N,Asparagine
Asp,D,Aspartic acid
Cys,C,Cysteine
Gln,Q,Glutamine
Glu,E,Glutamic acid
Gly,G,Glycine
His,H,Histidine
Ile,I,Isoleucine
Leu,L,Leucine
Lys,K,Lysine
Met,M,Methionine
Phe,F,Phenylalanine
Pro,P,Proline
Ser,S,Serine
Thr,T,Threonine
Trp,W,Tryptophan
Tyr,Y,Tyrosine
Val,V,Valine
Asx,B,Asparagine or Aspartic acid
Xle,J,Leucine or Isoleucine
Glx,Z,Glutamic acid or Glutamine
Xaa,X,Unknown
Unk,X,Unknown'''
reslet = dict()
for l in lettercode.split('\n'):
    l = l.split(',')
    reslet[l[0].upper()] = l[1].upper()
    reslet[l[1].upper()] = l[0].upper()
    reslet[l[0].lower()] = l[1].lower()
    reslet[l[1].lower()] = l[0].lower()
    reslet[l[0]] = l[1]
    reslet[l[1]] = l[0]

def toggle_code(sequence, direction='1to3'):
    result = ''
    if direction == '1to3':
        for char in sequence:
            result += reslet[char]
    else:
        for three in chunks(sequence, 3):
            result += reslet[three]
    return result

############
# BLOSUM62 #
############
#  Entropy =   0.6979, Expected =  -0.5209
b62h={a: i for i,a in enumerate(' A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  Z  X  * '.split())}
b62 = np.array([
[ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0,-2,-1, 0,-4],
[-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3,-1, 0,-1,-4],
[-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3, 3, 0,-1,-4],
[-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3, 4, 1,-1,-4],
[ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,-3,-3,-2,-4],
[-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2, 0, 3,-1,-4],
[-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2, 1, 4,-1,-4],
[ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3,-1,-2,-1,-4],
[-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3, 0, 0,-1,-4],
[-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3,-3,-3,-1,-4],
[-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1,-4,-3,-1,-4],
[-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2, 0, 1,-1,-4],
[-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1,-3,-1,-1,-4],
[-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1,-3,-3,-1,-4],
[-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2,-2,-1,-2,-4],
[ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2, 0, 0, 0,-4],
[ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0,-1,-1, 0,-4],
[-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3,-4,-3,-2,-4],
[-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1,-3,-2,-1,-4],
[ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4,-3,-2,-1,-4],
[-2,-1, 3, 4,-3, 0, 1,-1, 0,-3,-4, 0,-3,-3,-2, 0,-1,-4,-3,-3, 4, 1,-1,-4],
[-1, 0, 0, 1,-3, 3, 4,-2, 0,-3,-3, 1,-1,-3,-1, 0,-1,-3,-2,-2, 1, 4,-1,-4],
[ 0,-1,-1,-1,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2, 0, 0,-2,-1,-1,-1,-1,-1,-4],
[-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4, 1]])

def blosum(a,b):
   if not a.upper() in b62h:
       a = '*'
   if not b.upper() in b62h:
       b = '*'
   return b62[b62h[a.upper()],b62h[b.upper()]]
#############################################################################

def progress_bar(title, value, end, bar_width=50):
    ''' simplest progress bar '''
    percent = float(value) / end
    arrow = '-' * int(round(percent * bar_width)-1) + '>'
    spaces = ' ' * (bar_width - len(arrow))
    sys.stdout.write('\r{}: [{}] {}%'.format(title, arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()
    if percent==1.0:
        print()

def scale2(x):
    x -= x.mean(axis=0)
    x /= np.sqrt((x**2).sum()/(x.shape[0] - 1))
    return x

def eigenrank(atom_coordinates, numba=1):
    ''' calculating distance matrix -> adjacency matrices for different distance cutoffs -> PCA, scaling -> EigenRank '''
    if atom_coordinates.shape[1] != 3:
        atom_coordinates = atom_coordinates.T

    def pca(leaderranks):
        ''' PCA '''
        lrcenter = leaderranks - np.mean(leaderranks, axis=0)
        xt = lrcenter.T
        # xt = scale2(leaderranks).T
        cov_matrix = np.cov(xt)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        ind = np.argsort(-eigenvalues)# [::-1] -> minus instead
        princo = np.dot(eigenvectors.T, xt).T
        # fixing sign
        princo = princo * np.sign(eigenvectors[0,:])
        return scale2(princo[:,ind[0]])

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
            eg2 = np.dot(eg1,eg2)
        while error > error_threshold:
            M = eg2
            # eg2 = np.einsum('ij,j->i', eg1, eg2)
            eg2 = np.dot(eg1,eg2)
            error = np.sum(np.divide(np.abs(eg2-M),M))/(n+1)
            h+=1
        ground = eg2[n]/n
        leaderranks [:,a-5] = np.delete(eg2,-1)+ground

    return pca(leaderranks)



### ALIGNMENT CLASS ###
### numba part:
@jit(nopython=True, cache=args.no_cache)
def nlocalalign(ab,gap,factor,limit):
    m,n = ab.shape
    f = np.zeros((m+1,n+1))
    t = np.zeros((m+1,n+1))
    for i in range(1,m+1):
        prev = 0
        for j in range(1,n+1):
            down = f[i-1,j] - gap
            right = prev - gap
            diag = f[i-1,j-1] - factor*ab[i-1,j-1] + limit
            c = max([down,right,diag,0])
            prev,f[i,j] = c,c
            if diag == c:
                t[i,j] = 2
            elif right == c:
                t[i,j] = -1
            elif down == c:
                t[i,j] = 1
            else:
                t[i,j] = 0
    ## acquired f and t matrices, followed by traceback
    # plt.imshow(f)
    # cbar = plt.colorbar()
    # cbar.ax.tick_params(labelsize=18)
    # plt.xlabel('residue index', fontsize=18)
    # plt.ylabel('residue index', fontsize=18)
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    # # plt.savefig('.svg', dpi=1200)
    # plt.show()
    # plt.close()
    i,j = np.where(f==f.max())
    i,j = i[0],j[0]
    score = f[i,j]
    is_gap = []
    i_list = []
    j_list = []
    c = 0
    while t[i,j] != 0:
        is_gap.append(1)
        i_list.append(0)
        j_list.append(0)
        i_list[c] = i-1
        j_list[c] = j-1

        editing = t[i,j]
        if editing != 2:
            if editing == -1:
                j -= 1
                is_gap[c] = 1
            else:
                i -= 1
                is_gap[c] = 2
        else:
            is_gap[c] = 0
            i -= 1
            j -= 1
        c += 1
    return np.array(i_list, dtype=np.int32),np.array(j_list, dtype=np.int32),np.array(is_gap, dtype=np.int32),score

class Alignment:
    '''pairwise alignment of two protein structures.
    query and target as Protein objects.'''
    def __init__(self, query, target, gap, limit, factor=1):
        self.query = query
        self.target = target
        self.limit = float(limit)
        self.factor = float(factor)
        self.gap = float(gap)
        self.align_and_rotate()

    def align_and_rotate(self): # get the local alignment, calculate optimal rotation matrix for structures to fit into each other
        ab = dm_euclidian(self.query.er, self.target.er) # normal distribution (dmnd) or difference (dm_euclidian)
        # plt.imshow(ab) 
        # cbar = plt.colorbar()
        # cbar.ax.tick_params(labelsize=18)
        # plt.xlabel('residue index', fontsize=18)
        # plt.ylabel('residue index', fontsize=18)
        # plt.xticks(fontsize=18)
        # plt.yticks(fontsize=18)
        # # plt.savefig('.svg', dpi=1200)
        # plt.show()

        # actual alignment, using the fast SW from above
        self.i_list, self.j_list, self.is_gap, self.score = nlocalalign(ab,self.gap,self.factor,self.limit)

        self.traceback_len = len(self.is_gap)

        i_list = [i for i,g in zip(self.i_list,self.is_gap) if g == 0]
        j_list = [j for j,g in zip(self.j_list,self.is_gap) if g == 0]
        self.len_wo_gaps = len(i_list)
        # if (len(i_list) < 10 or len(j_list) < 10):
        #     print(self.query.id, self.target.id, self.gap, self.limit)
        self.nrgaps = np.count_nonzero(self.is_gap)

        # Kabsch
        a_pre = self.query.coordinates
        b_pre= self.target.coordinates
        a = a_pre[i_list,:]
        b = b_pre[j_list,:]
        self.query_centroid = np.mean(a, axis=0)
        self.target_centroid = np.mean(b, axis=0)
        a -= self.query_centroid
        b -= self.target_centroid
        h = a.T@b
        u,s,v = np.linalg.svd(h.T)
        d = np.linalg.det(v.T@u.T)
        r = v.T@np.diag([1,1,d])@u.T
        a = a@r
        self.rmsd = rmsd(a,b)
        self.rotation_matrix = r
        self.query_aligned = a
        self.target_aligned = b
        self.dists = np.linalg.norm(a-b, axis=1)
        # print('query centroid', self.query_centroid)
        # print('target centroid', self.target_centroid)
        # print('rotation', r)

        # GDT_TS:
        f1 = np.count_nonzero(np.where(self.dists < 1))
        f2 = np.count_nonzero(np.where(self.dists < 2))
        f4 = np.count_nonzero(np.where(self.dists < 4))
        f8 = np.count_nonzero(np.where(self.dists < 8))
        self.gdt_ts = 25 * sum([f1,f2,f4,f8])/self.len_wo_gaps if self.len_wo_gaps > 0 else 0
        # FATCAT-inspired similarity score
        ###########################################################################
        #GDT-sim "improved", needs further tinkering...
        self.gdt_sim = self.score * self.len_wo_gaps*self.gdt_ts

        # TMscore
        d0 = 1.24*np.cbrt(self.target.l - 15) -1.8
        di = np.sqrt(np.sum((a-b)**2, axis=1))
        self.tmq = np.sum(1/(1+(di/d0)**2))/self.query.l
        self.tmt = np.sum(1/(1+(di/d0)**2))/self.target.l
        self.tm = (self.tmq+self.tmt)/2



#######################################################################
# STRUCTURE (coordinates, EigenRank profile, ID)

class structure:
    def __init__(self, file):
        self.file = file
        self.id = os.path.basename(file).split('.')[0][:]
        self.sccs = ''
        self.parse_coords(file)
        self.l = self.coordinates.shape[0]
        if self.l < 10:
            print(self.file, ' too short for sensible EigenRank: ', self.l)
        else:
            self.er = eigenrank(self.coordinates)

    def parse_coords(self,file):
        atom_id = -20
        res_id = -20
        # no HIS tags etc....or?
        atoms = []
        # STUPID PDB
        for line in open(file, 'r'):
            if (line.startswith('ATOM') and
                    # res_id < int(line[22:26]) and
                    # atom_id < int(line[5:11]) and
                    line[13:15] == 'CA' and
                    line[16] in ['A',' ']):
                atom_id = int(line[5:11])
                x,y,z = float(line[30:38]),float(line[38:46]),float(line[46:54])
                res = line[17:20]
                res_id = int(line[22:26])
                atoms.append({'res_id':res_id, 'res':res, 'atom_id':atom_id, 'coords':[x,y,z]})
            elif line.startswith('REMARK  99 ASTRAL SCOPe-sccs:'):
                self.sccs = line[30:].rstrip()
            elif line.startswith('REMARK  99 ASTRAL SCOPe-sid:'):
                self.sid = line[29:].rstrip()
            elif line.startswith('ENDMDL'):
                break
        self.atoms = atoms
        self.coordinates = np.asarray([a['coords'] for a in atoms])

### OUTPUT FILE
if args.output != False:
    output_file = open(args.output, 'w')
## TESTING
# input1 = ['../databases/COPS/queries/pdb/c2nn6C_.pdb']
# input2 = ['../databases/COPS/database/pdb/c2c39T_.pdb']
# args.gap_cost=0.5
# args.limit=1.4
# args.verbose=True
# args.multiprocessing=True
# args.no_banner=True
### INITIATING STRUCTURE OBJECTS
t4 = time()
if args.multiprocessing:
    n_cores = mp.cpu_count()
    with mp.Pool(n_cores) as pool:
        structures1 = pool.map(structure, input1)
        t5 = time()
        structures2 = pool.map(structure, input2)
else:
    structures1 = [structure(s1) for s1 in input1]
    t5 = time()
    structures2 = [structure(s2) for s2 in input2]
if args.verbose:
    print('# EigenRank calculation time: {0:.3f}s + {1:.3f}s = {2:.3f}s'.format(t5-t4, time()-t5, time()-t4))
    nstr1=len(structures1)
    nstr2=len(structures2)
    total_number_of_alignments = nstr1*nstr2
    if args.output == False:
        print('# EigenRank calculation time: {0:.3f}s + {1:.3f}s = {2:.3f}s'.format(t5-t4, time()-t5, time()-t4))
        print('# input1: {0} structures, input2: {1} structures'.format(nstr1,nstr2))
        print('# starting pairwise alignment (gap cost = {0}, limit = {1})\n'.format(args.gap_cost, args.limit))
    else:
        output_file.write('# EigenRank calculation time: {0:.3f}s + {1:.3f}s = {2:.3f}s\n'.format(t5-t4, time()-t5, time()-t4))
        output_file.write('# input1: {0} structures, input2: {1} structures\n'.format(nstr1,nstr2))
        output_file.write('# starting pairwise alignment (gap cost = {0} limit = {1} gap2 = {2} limit2 = {3} )\n'.format(args.gap_cost, args.limit, args.gap2, args.limit2))
        output_file.flush()
# with open('structure_COPS.pkl', 'wb') as p:
#     pickle.dump(qstructures, p)
#     pickle.dump(tstructures, p)

# with open('structure_COPS.pkl', 'rb') as p:
    # qstructures=pickle.load(p)
    # tstructures=pickle.load(p)
def sauzer_mp(queue,q,t,gap2=args.gap2,limit2=args.limit2,gap=args.gap_cost,limit=args.limit):
    ''' generating output for alignment '''
    ali = Alignment(q,t,gap,limit)
    normi = 0
    if ali.tm > 0.1:
        qrotated = (q.coordinates-ali.query_centroid)@ali.rotation_matrix
        trotated = (t.coordinates-ali.target_centroid)
        # with open(q.id+'.xyz', 'w') as f:
        #     f.write('\n'.join(['C {0:.3f} {1:.3f} {2:.3f}'.format(x,y,z) for x,y,z in qrotated]))
        # with open(t.id+'.xyz', 'w') as f:
        #     f.write('\n'.join(['C {0:.3f} {1:.3f} {2:.3f}'.format(x,y,z) for x,y,z in trotated]))

        dm = spatial.distance_matrix(qrotated, trotated, p=2)
        # plt.imshow(dm)
        # cbar = plt.colorbar()
        # cbar.ax.tick_params(labelsize=18)
        # plt.xlabel('residue index', fontsize=18)
        # plt.ylabel('residue index', fontsize=18)
        # plt.xticks(fontsize=18)
        # plt.yticks(fontsize=18)
        # # plt.savefig('.svg', dpi=1200)
        # plt.show()

        i_list, j_list, is_gap, score = nlocalalign(dm,gap2,limit2,1)

        traceback_len = len(is_gap)

        i_list = [i for i,g in zip(i_list,is_gap) if g == 0]
        j_list = [j for j,g in zip(j_list,is_gap) if g == 0]
        len_wo_gaps = len(i_list)
        nrgaps = np.count_nonzero(is_gap)

        if not (len(i_list) < 10 or len(j_list) < 10):

            # Kabsch
            a_pre = qrotated
            b_pre = trotated
            a = a_pre[i_list,:]
            b = b_pre[j_list,:]
            query_centroid = np.mean(a, axis=0)
            target_centroid = np.mean(b, axis=0)
            a -= query_centroid
            b -= target_centroid
            h = a.T@b
            u,s,v = np.linalg.svd(h.T)
            d = np.linalg.det(v.T@u.T)
            r = v.T@np.diag([1,1,d])@u.T
            a = a@r
            # rmsd = rmsd(a,b)
            rotation_matrix = r
            query_agned = a
            target_agned = b
            dists = np.linalg.norm(a-b, axis=1)
            print('query centroid', query_centroid)
            print('target centroid', target_centroid)
            print('rotation', r)

            # GDT_TS:
            f1 = np.count_nonzero(np.where(dists < 1))
            f2 = np.count_nonzero(np.where(dists < 2))
            f4 = np.count_nonzero(np.where(dists < 4))
            f8 = np.count_nonzero(np.where(dists < 8))
            gdt_ts = 25 * sum([f1,f2,f4,f8])/len_wo_gaps if len_wo_gaps > 0 else 0
            # FATCAT-inspired similarity score
            ###########################################################################
            #GDT-sim "improved", needs further tinkering...
            gdt_sim = score * len_wo_gaps*gdt_ts

            # TMscore
            d0 = 1.24*np.cbrt(t.l - 15) -1.8
            di = np.sqrt(np.sum((a-b)**2, axis=1))
            tmq = np.sum(1/(1+(di/d0)**2))/q.l
            tmt = np.sum(1/(1+(di/d0)**2))/t.l
            tm = (tmq+tmt)/2
            if tm > ali.tm:
                qrotated_ref = (qrotated-query_centroid)@rotation_matrix
                trotated_ref = (trotated-target_centroid)
                with open(q.id+'2.xyz', 'w') as f:
                    f.write('\n'.join(['C {0:.3f} {1:.3f} {2:.3f}'.format(x,y,z) for x,y,z in qrotated_ref]))
                with open(t.id+'2.xyz', 'w') as f:
                    f.write('\n'.join(['C {0:.3f} {1:.3f} {2:.3f}'.format(x,y,z) for x,y,z in trotated_ref]))

                chain_1 = ''.join([toggle_code(q.atoms[i]['res'],'3to1') if (g!=1) else '-' for i,g in zip(i_list,is_gap)])[::-1]
                chain_2 = ''.join([toggle_code(t.atoms[j]['res'],'3to1') if (g!=2) else '-' for j,g in zip(j_list,is_gap)])[::-1]
                if args.no_banner:
                    output = []
                else:
                    output = ['', *logo, '']
                output.append(' '.join(['chain_1:','{:4d}'.format(q.l), str(q.file), q.sccs]))
                output.append(' '.join(['chain_2:','{:4d}'.format(t.l), str(t.file), t.sccs]))
                output.append('')
                output.append(' '.join(['alignment_length:', '{:d}'.format(traceback_len),
                    'gaps:', '{0:d} ({1:.2%})'.format(nrgaps,nrgaps/traceback_len),
                    # 'RMSD:', '{:.3f}'.format(rmsd),
                    '\nGDT-TS:', '{:.3f}'.format(gdt_ts),
                    'GDT-sim:', '{:.3f}'.format(gdt_sim),
                    'Zerscore:', '{:.3f}'.format(score),
                    '\nTMscore (normased by length of chain_1):', '{:.5f}'.format(tmq),
                    '\nTMscore (normased by length of chain_2):', '{:.5f}'.format(tmt),
                    '\nidentity:', '{:.3f}'.format(np.count_nonzero([bool(a==b) for a,b in zip(chain_1,chain_2)])/traceback_len),'similarity:','{:.3f}'.format(np.count_nonzero([bool(blosum(a,b)>0) for a,b in zip(chain_1,chain_2)])/traceback_len)]))
                output.append('')
                output.append(chain_1)
                output.append(''.join([':' if (g==0 and d<5) else ' ' for d,g in zip(dists ,is_gap)]))
                output.append(chain_2)
                output.append('')
                output = '\n'.join(output)
                if args.output == False:
                    print(output)
                else:
                    queue.put(output)
                return True
            else:
                normi = 1
        else:
            normi=1
    else:
        normi=1
# normal
    if normi==1:
        chain_1 = ''.join([toggle_code(q.atoms[i]['res'],'3to1') if (g!=1) else '-' for i,g in zip(ali.i_list,ali.is_gap)])[::-1]
        chain_2 = ''.join([toggle_code(t.atoms[j]['res'],'3to1') if (g!=2) else '-' for j,g in zip(ali.j_list,ali.is_gap)])[::-1]
        if args.no_banner:
            output = []
        else:
            output = ['', *logo, '']
        output.append(' '.join(['chain_1:','{:4d}'.format(q.l), str(q.file), q.sccs]))
        output.append(' '.join(['chain_2:','{:4d}'.format(t.l), str(t.file), t.sccs]))
        output.append('')
        output.append(' '.join(['alignment_length:', '{:d}'.format(ali.traceback_len),
            'gaps:', '{0:d} ({1:.2%})'.format(ali.nrgaps,ali.nrgaps/ali.traceback_len),
            # 'RMSD:', '{:.3f}'.format(ali.rmsd),
            '\nGDT-TS:', '{:.3f}'.format(ali.gdt_ts),
            'GDT-sim:', '{:.3f}'.format(ali.gdt_sim),
            'Zerscore:', '{:.3f}'.format(ali.score),
            '\nTMscore (normased by length of chain_1):', '{:.5f}'.format(ali.tmq),
            '\nTMscore (normased by length of chain_2):', '{:.5f}'.format(ali.tmt),
            '\nidentity:', '{:.3f}'.format(np.count_nonzero([bool(a==b) for a,b in zip(chain_1,chain_2)])/ali.traceback_len),'similarity:','{:.3f}'.format(np.count_nonzero([bool(blosum(a,b)>0) for a,b in zip(chain_1,chain_2)])/ali.traceback_len)]))
        output.append('')
        output.append(chain_1)
        output.append(''.join([':' if (g==0 and d<5) else ' ' for d,g in zip(ali.dists ,ali.is_gap)]))
        output.append(chain_2)
        output.append('')
        output = '\n'.join(output)
        if args.output == False:
            print(output)
        else:
            queue.put(output)
        return True


def writing2output(q):
    while 1:
        m = q.get()
        if m == 'kill':
            break
        output_file.write('\n'+m)
        output_file.flush()


# atom_id = int(line[5:11])
# x,y,z = float(line[30:38]),float(line[38:46]),float(line[46:54])
# res = line[17:20]
# res_id = int(line[22:26])
    # with open(q.id+'_'+t.id+'.pdb', 'w') as f:
    #     ai = [q.atoms[i]['atom_id'] for i in a.i_list]
    #     r = [q.atoms[i]['res'] for i in a.i_list]
    #     ri = [q.atoms[i]['res_id'] for i in a.i_list]
    #     x,y,z = (a.query_aligned[:,i] for i in [0,1,2])
    #     f.write(backtopdb('A',ai,r,ri,x,y,z))
    #     ai = [t.atoms[i]['atom_id'] for i in a.j_list]
    #     r = [t.atoms[i]['res'] for i in a.j_list]
    #     ri = [t.atoms[i]['res_id'] for i in a.j_list]
    #     x,y,z = (a.target_aligned[:,i] for i in [0,1,2])
    #     f.write(backtopdb('B',ai,r,ri,x,y,z))
def backtopdb(chain,ai,r,ri,x,y,z):
    return '\n'.join(['ATOM  {0:5d}  CA  {1} {6}{2:4d}    {3:8.3f}{4:8.3f}{5:8.3f}'.format(ai_,r_,ri_,x_,y_,z_,chain) for ai_,r_,ri_,x_,y_,z_ in zip(ai,r,ri,x,y,z)]+['TER', ''])
la = 'ATOM     15  CA  ILE A   3       0.995  10.450  37.068  1.00  0.04           C'

### PAIRWISE ALIGNMENTS
# (for now merely printing to stdout; append '> resultfile.txt' at the end of your command)
t3 = time()
# gap2s = np.linspace(0.2, 2, 19)
# limit2s = np.linspace(0.2, 2, 19)
# gaps = np.linspace(0.2, 2, 19)
# limits = np.linspace(0.2, 2, 19)
# co = itertools.product(structures1, structures2, gap2s, limit2s, gaps, limits)
if args.multiprocessing:
    n_cores = mp.cpu_count()
    manager = mp.Manager()
    queue = manager.Queue()
    co = itertools.product([queue], structures1, structures2)
    with mp.Pool(n_cores) as pool:
        filewriter = pool.apply_async(writing2output, (queue,))
        outputs = pool.starmap(sauzer_mp, co)
        queue.put('kill')
else:
    co = itertools.product(structures1, structures2)
    outputs = [sauzer(q,t) for q,t in co]
if args.verbose:
    t4=time()-t3
    if args.output == False:
        print('# alignment time: {0:.3f}s, {1:.3f}ms/aligment'.format(t4, t4/total_number_of_alignments*1000))
    else:
        output_file.write('# alignment time: {0:.3f}s, {1:.3f}ms/aligment'.format(t4, t4/total_number_of_alignments*1000))
if args.output != False:
    output_file.close()
sys.exit()

### COMPRESSION OF RESULTS
txz = tarfile.open(time.strftime('%Y%m%d_%H%M%S_sauzeror.txz'), 'w|xz')
for i,(o,q,t) in enumerate(outputs):
    by = o.encode()
    info = tarfile.TarInfo(name=q+'/'+t+'.txt')
    info.size=len(by)
    txz.addfile(info, fileobj=io.BytesIO(by))
txz.close()
print(time()-start)

sys.exit()

def scope_hierarchy():
    ''' building a SCOPe hierarchy dict structure '''
    scope = {}
    scope_id_set = set()
    # class -> fold -> superfamily -> family
    for line in open(scope_cla, 'r'):
        if not line.startswith('#'):
            line = line.lower().split('\t')
            sid = line[0]
            cid = line[3].split('.') # class ID
            scope_id_set.add(sid)
            if cid[0] in scope:
                if cid[1] in scope[cid[0]]:
                    if cid[2] in scope[cid[0]][cid[1]]:
                        if cid[3] in scope[cid[0]][cid[1]][cid[2]]:
                            scope[cid[0]][cid[1]][cid[2]][cid[3]].add(sid)
                        else:
                            scope[cid[0]][cid[1]][cid[2]][cid[3]] = {sid}
                    else:
                        scope[cid[0]][cid[1]][cid[2]] = {}
                else:
                    scope[cid[0]][cid[1]] = {}
            else: scope[cid[0]] = {}
    return scope
