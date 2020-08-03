import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,json,re,sys,io
from pathlib import Path
from time import time
import pickle
import tarfile


# read result files
files = ['/home/julius/biotech/sauzeror/listing_no-head.txt']
listing = dict()
for line in open(files[0],'r'):
    listing[line[:6]] = list(line[7:].rstrip().split('\t'))


def roc(datadict):
    sorted_scores = {query[1:]: [t[2][1:] for t in targets] for query,targets in datadict.items()}
    enum_query = {query: i for i,query in enumerate(sorted_scores.keys())}
    ids = np.empty((176,1056), dtype='U6')
    for query,v in sorted_scores.items():
        ids[enum_query[query],:] = v
    results = np.zeros((176,4,6), dtype=int)
    ranks = np.squeeze(np.asarray([[np.where(h == ids[i,:]) for h in listing[query]] for query,i in enum_query.items()]))
    for query,i in enum_query.items():
        for k in range(6):
            # 0:TP, 1:FP, 2:TN, 3:FN
            results[i,0,k] = np.sum(np.less_equal(ranks[i,:], k))
            results[i,1,k] = k+1 - results[i,0,k]
            results[i,3,k] = 6 - results[i,0,k]
            results[i,2,k] = 1056 - np.sum(results[i,[0,1,3],k])
            # results[i,1,k] = np.sum(np.greater(ranks[i,:k+1], k))
            # results[i,3,k] = sum([t in listing[query] for t in ids[i,k+1:]])
    tp = np.sum(results[:,0,:], axis=0)
    fp = np.sum(results[:,1,:], axis=0)
    tn = np.sum(results[:,2,:], axis=0)
    fn = np.sum(results[:,3,:], axis=0)
    # print((tp+fp)/176)
    # print(tp,fp,fn,tn)
    # print(tp+fp+fn+tn)
    tpr = tp/(tp+fn)
    fpr = fp/1056
    dgs = np.sqrt(1/tpr.shape[0] * np.sum((np.linspace(1/6,1,6)-tpr)**2+(np.zeros(6)-fpr)**2))
    return list(tpr), list(fpr), dgs

# tpr, fpr = roc(cops)

def devgoldstand(tpr,fpr):
    dgs = np.sqrt(1/tpr.shape[0] * np.sum((np.linspace(1/6,1,6)-tpr)**2+(np.zeros(6)-fpr)**2))
    return dgs
# plt.plot(fpr,tpr)
# plt.show()
# print(devgoldstand(tpr,fpr))

t2 = time()
def read_onefile(result_file):
    scores = dict()
    for line in open(result_file, 'r'):
        if not line.startswith('#'):
            if line.startswith('chain_1'):
                c1 = os.path.basename(line.split()[2]).split('.')[0]
                l1 = int(line.split()[1])
            if line.startswith('chain_2'):
                c2 = os.path.basename(line.split()[2]).split('.')[0]
                l2 = int(line.split()[1])
            if line.startswith('alignment_length:'):
                tracebacklen = int(line.split()[1])
                nrgaps = int(line.split()[3])
            if line.lower().startswith('gdt-ts:'):
                gdtts = float(line.split()[1])
                gdtsim = float(line.split()[3])
                zerscore = float(line.split()[5])
            if line.lower().startswith('tmscore (normalised by length of chain_1):'):
                tm1 = float(line.split()[6])
            if line.lower().startswith('tmscore (normalised by length of chain_2):'):
                tm2 = float(line.split()[6])
                tm = (tm1+tm2)/2
                sauze = tm*zerscore**2*(tracebacklen-nrgaps)**2*gdtts**2*l1**-0.5*l2**-0.5
                if c1 in scores:
                    if not (tm, tm1, c2) in scores[c1]:
                        scores[c1].append((tm, tm1, c2))
                else:
                    scores[c1] = [(tm, tm1, c2)]
    sauzers = {c1:sorted(scores[c1], reverse=True, key=lambda x: x[0]) for c1 in scores.keys()}
    return sauzers#, (g,l,g2,l2)


# after = read_onefile('COPS_after.txt')
# all = {t[2] for t in after['c2nn6C_']}
# for c1,c2s in after.items():
#     if len(c2s) < 1056:
#         print(len(c2s))
#         missing = {c2m[2] for c2m in c2s if c2m[2] not in all}
#         print(c1, missing)
# print(roc(read_onefile('COPS_before.txt')), sep='\n')
# print(roc(read_onefile('COPS_after.txt')), sep='\n')
# print(roc(read_onefile('COPSref_003.txt')), sep='\n')
# print(time()-t2)
# sys.exit()
# with open('storage02.pkl', 'rb') as f:
#     ces = pickle.load(f)
#     tmaligns = pickle.load(f)
#     fatcats = pickle.load(f)
#     tmaligns_val = pickle.load(f)

# sss = []
# for i,d in enumerate(Path('./COPSref/').glob('COPS*.txt')):
#     j,pa = read_onefile(d)
#     sss.append((len([j[h] for h in j.keys() if (sum([g[1] for g in j[h] if g[1]<0.4])>0)]),*pa,d))

# print(*sorted(sss), sep='\n')
# sys.exit()

fig,ax = plt.subplots()
dgs = []
# for i,d in enumerate(Path('./COPSref/').glob('COPS*.txt')):
# for i,d in enumerate(['bestparams.txt', 'COPS_sauzeror_to10ang_0.8_0.7.txt', 'COPS_sauzeror_0.8_0.7.txt', 'COPS_normal.txt', '4COPS_new2_params.txt', 'proof_bestparams.txt', '4COPS_no_scaling.txt', '4COPS_not_even_pca.txt']):
# for i,d in enumerate(sorted(Path('./').glob('*COPS-08-07.txt'))):
# for i,d in enumerate(['only4LRCOPS.txt', 'scaledLR-COPS.txt', 'COPS_normal.txt', 'nonscaledLR-COPS.txt']):
for i,d in enumerate(['scaled_COPS.txt', 'notscaled_COPS.txt', 'vanillaLR_COPS.txt', 'COPS-refinement.txt']):
# for d in ['COPS-refinement.txt']:
    print(d)
    r = roc(read_onefile(d))
    print('tpr:{}\nfpr:{}\ndgs:{}'.format(*r))
    ax.plot(r[1],r[0], label='{0} (DGS: {1:.4f})'.format(str(d)[:-4],r[2]))
ax.set_ylim(ymin=0)
ax.set_xlim(xmin=0)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
ax.legend()
plt.show()
# plt.savefig('ROC_COPS_PCA_questionmark_;)_sauze.png', dpi=600)
sys.exit()


#######################################################################




# data_dir = Path('/home/julius/Sync/cops_results/')
# data_files = data_dir.glob('*.txt')
# droc = {}
# for f in data_files:
#     tpr,fpr = read_cops(f)
#     if tpr[5] > 0.85:
#         droc[str(f)[-11:-4]] = (tpr,fpr)

# with open('roc.pkl', 'wb') as f:
#     pickle.dump(droc, f)
# with open('roc.pkl', 'rb') as f:
#     droc = pickle.load(f)
# # droc = {k: v for k,v in droc.items() if v[0][5]>0.8}

# factor,limit,dgs = [],[],[]
# for k,v in droc.items():
#     factor.append(float(k[:3]))
#     limit.append(float(k[4:]))
#     dgs.append(devgoldstand(v[0],v[1]))
# print(factor, len(limit), len(dgs))

# # print(len(droc.items()))
# # print(droc.keys())
# # for k,v in droc.items():
#    #  x,y = v[1],v[0]
# #     plt.plot(x,y, '-o')

# # plt.show()
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# import numpy as np


# fig = plt.figure()
# ax = fig.gca(projection='3d')

# # X = np.arange(-5, 5, 0.25)
# # Y = np.arange(-5, 5, 0.25)
# # X, Y = np.meshgrid(X, Y)
# factor, limit = np.meshgrid(factor, limit)

# # R = np.sqrt(X**2 + Y**2)
# # Z = np.sin(R)
# # Plot the surface.
# surf = ax.plot_surface(factor, limit, dgs, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)

# # Customize the z axis.
# # ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()
