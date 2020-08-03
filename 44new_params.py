#!/usr/bin/python
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
#               ║  │   --gap-cost     gap cost parameter (default 0.8)    │                ║
#               ║  │   --limit        limit parameter (default 0.7)       │                ║
#               ║  │                                                      │                ║
#               ╟──┤   -o  --output     output file                       │                ║
#               ║  └──────────────────────────────────────────────────────┘                ║
#               ║  ┌───────────────────────────────────────────────────────────────────┐   ║
#               ╟──┤ further options may be given instead of [1..]                     │   ║
#               ║  │   -h   --help              show this help message                 │   ║
#               ║  │   -v   --verbose           for more verbosity (starting with #)   │   ║
#               ║  │   -a   --atomium           use atomium parser                     │   ║
#               ║  │        --csv               csv output for easy parsing;           │   ║
#               ║  │                            run with --csv alone to show header    │   ║
#               ╟──┤   -nc  --no-cache          don't cache numbas machine code        │   ║
#               ║  └───────────────────────────────────────────────────────────────────┘   ║
#               ╚══════════════════════════════════════════════════════════════════════════╝
from time import time
from time import strftime
import numpy as np
# numba is crucial for alignment... takes far too long otherwise
from numba import jit, float32, int32, types
from scipy import spatial
from scipy import stats
from scipy.special import ndtr
import sys,os,pickle
import tarfile, io
import gzip
from pathlib import Path
import itertools
import argparse
import multiprocessing as mp
import matplotlib.pyplot as plt
# atomium is optional, but all other packages should be installed
try:
    import atomium
    has_atomium = True
except ImportError:
    has_atomium = False

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# term_width = os.get_terminal_size().columns
logo = []
for i,line in enumerate(open(__file__, 'r')):
    if 2 <= i <= 8:
        logo.append((line[2:-2]))
helptext = []
for i,line in enumerate(open(__file__, 'r')):
    if (i>0 and line.startswith('#')):
        helptext.append((line[1:].rstrip()))
    elif not line.startswith('#'):
        break

### ARGPARSE ROUTINE

parser = argparse.ArgumentParser('SAUZEROR', add_help=False)
subparser = parser.add_subparsers()

parser_a = subparser.add_parser('align', help='align sets of protein structures pairwise')
parser_a.add_argument('input1', type=str, help='first input')
parser_a.add_argument('input2', type=str, help='second input')
parser_a.add_argument('--gap-cost', type=float, default=0.8, help='gap cost')
parser_a.add_argument('--limit', type=float, default=0.7, help='limit parameter for Smith-Waterman-algorithm')
parser_a.add_argument('-o', '--output', default=False, help='output to file instead of stdout')

# future feature
parser_c = subparser.add_parser('classify', help='classify a set of proteins with SCOPe')
parser_c.add_argument('input', type=str, help='input structure')

parser.add_argument('-v', '--verbose', default=False, action='store_true', help='verbose output')
parser.add_argument('-a', '--atomium', default=False, action='store_true', help='atomium parser')
parser.add_argument('--csv', default=False, action='store_true', help='output csv for easy parsing')
parser.add_argument('-nc', '--no-cache', action='store_false', default=True, help='DON\'T write numba machine code to cache')

parser.add_argument('-h', '--help', action='store_true', help='that\'s better')
args = parser.parse_args()

if args.help:
    print('\n'.join(helptext))

if (has_atomium and args.atomium):
    use_atomium = True
elif (not has_atomium and args.atomium):
    print('to use this feature, please install atomium: \'pip install atomium\'')
    use_atomium = False
else:
    use_atomium = False

if hasattr(args,'input'):
    print('classification of ', args.input)
    sys.exit()
elif hasattr(args,'input1'):
    inputs = dict()
    for i,inargs in enumerate([args.input1, args.input2]):
        if os.path.isdir(inargs):
            inputs['input{}'.format(i)] = list(Path(inargs).rglob('*.pdb'))
            inputs['input{}'.format(i)].extend(Path(inargs).rglob('*.ent'))
            inputs['input{}'.format(i)].extend(Path(inargs).rglob('*.cif'))
            inputs['input{}'.format(i)].extend(Path(inargs).rglob('*.atm'))
            inputs['input{}'.format(i)].extend(Path(inargs).rglob('*.gz'))
        elif os.path.isfile(inargs):
            inputs['input{}'.format(i)] = []
            if 'gz' in str(inargs):
                inputs['input{}'.format(i)] = [os.path.abspath(inargs)]
            else:
                for line in open(inargs,'r'):
                    line=line.rstrip()
                    if not line.startswith('#'):
                        if not os.path.isfile(line):
                            inputs['input{}'.format(i)] = [os.path.abspath(inargs)]
                            break
                        else:
                            inputs['input{}'.format(i)].append(os.path.abspath(line.rstrip()))
else:
    if args.csv:
        print('chain_1,length_1,sccs_1,chain_2,length_2,sccs_2,ali_length,gaps,gaps_percent,rmsd,t_sauze,gdt_ts,gdt_similarity,zer_score,tm_score1,tm_score2,identity,similarity\naligned chain sequence 1\nindication of < 5 Å\naligned chain sequence 2')
        sys.exit()
    if not args.help:
        print('\n'.join(helptext))
    sys.exit()
### GENERAL FUNCTIONS

def dm_euclidian(a,b):
    ''' euclidian distance matrix of 2 sequences '''
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
#############################################################

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
        return scale2(princo[:,ind[:2]])

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
        # ab = dm_euclidian(self.query.er, self.target.er) # normal distribution (dmnd) or difference (dm_euclidian)
        ab = spatial.distance_matrix(self.query.er, self.target.er)

        # actual alignment, using the fast SW from above
        self.i_list, self.j_list, self.is_gap, self.score = nlocalalign(ab,self.gap,self.factor,self.limit)

        self.traceback_len = len(self.is_gap)

        i_list = [i for i,g in zip(self.i_list,self.is_gap) if g == 0]
        j_list = [j for j,g in zip(self.j_list,self.is_gap) if g == 0]
        self.len_wo_gaps = len(i_list)
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

        # ridiculous score that gives the best results in the COPS benchmark
        self.t_sauze = self.tm*self.score**2*self.traceback_len**2*self.gdt_ts**2*self.query.l**-0.5*self.target.l**-0.5



#######################################################################
# STRUCTURE (coordinates, EigenRank profile, ID)

class structure:
    def __init__(self, file):
        self.file = file
        self.sccs = ' '
        self.atoms = []
        self.id = os.path.basename(file).split('.')[0]
        self.parse_coords(file)

    def parse_coords(self,file):
        atom_id = -20
        res_id = -20
        # no HIS tags etc....or?
        # STUPID PDB
        if '.gz' in str(file):
            foe = gzip.open(file,'rt')
        else:
            foe = open(file, 'r')
        for line in foe:
            if (line.startswith('ATOM') and
                    # res_id < int(line[22:26]) and   # in case there's misbehaviour with faulty 
                    # atom_id < int(line[5:11]) and   # .pdb files --> allows only "right" order
                    line[13:15] == 'CA' and
                    line[16] in ['A',' ']):           # take only first alternative
                atom_id = int(line[5:11])
                x,y,z = float(line[30:38]),float(line[38:46]),float(line[46:54])
                res = line[17:20]
                res_id = int(line[22:26])
                self.atoms.append({'res_id':res_id, 'res':res, 'atom_id':atom_id, 'coords':[x,y,z]})
            elif line.startswith('REMARK  99 ASTRAL SCOPe-sccs:'):  # \
                self.sccs = line[30:].rstrip()                      # | if at all you're interested
            elif line.startswith('REMARK  99 ASTRAL SCOPe-sid:'):   # | to know SCOPe's IDs
                self.sid = line[29:].rstrip()                       # /
            # stop parsing after first chain 
            # use --atomium and read structure.atoms for all chains
            elif (line.startswith('ENDMDL') or line.startswith('TER')):
                break
        self.coordinates = np.asarray([a['coords'] for a in self.atoms])
        self.l = self.coordinates.shape[0]
        if self.l < 10:
            print('{} is too short for a sensible EigenRank ({})'.format(self.id, self.l))
        else:
            self.er = eigenrank(self.coordinates)


### OUTPUT FILE
if args.output != False:
    output_file = open(args.output, 'w')
### INITIATING STRUCTURE OBJECTS
t4 = time()
# not too elegant... but easier than calling the dictionary in every line
input1 = inputs['input0']
input2 = inputs['input1']
# automatically fire up all processors which can take up to a second
if len(input1)+len(input2) >= 20:
    n_cores = mp.cpu_count()
    with mp.Pool(n_cores) as pool:
        structures1 = pool.map(structure, input1)
        t5 = time()
        structures2 = pool.map(structure, input2)
else:
    structures1 = [structure(s1) for s1 in input1]
    t5 = time()
    structures2 = [structure(s2) for s2 in input2]



### OUTPUT GENERATOR
def sauzer(queue,q,t,gap=args.gap_cost,limit=args.limit):
    if not (hasattr(q, 'er') and hasattr(t, 'er')):
        return None
    a = Alignment(q,t,gap,limit)
    output = '{0:.3f} {1:.3f} {2:.5f} {3:.5f} {4:3d}'.format(gap,limit,a.tmq,a.tmt, a.traceback_len)
    queue.put(output)
    return None

# to safely write files while multiprocessing
def writing2output(q):
    while 1:
        m = q.get()
        if m == 'kill':
            break
        output_file.write('\n'+m)
        output_file.flush()

### PAIRWISE ALIGNMENTS
t3 = time()
gaps = np.linspace(0.5, 1, 20)
limits = np.linspace(1.8, 3, 20)

n_cores = mp.cpu_count()
manager = mp.Manager()
queue = manager.Queue()
co = itertools.product([queue], structures1, structures2, gaps, limits)
with mp.Pool(n_cores) as pool:
    filewriter = pool.apply_async(writing2output, (queue,))
    outputs = pool.starmap(sauzer, co)
    queue.put('kill')

if args.output != False:
    output_file.close()
sys.exit()

