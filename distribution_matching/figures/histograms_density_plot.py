
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import numpy as np

from data import LabelledCollection

scale = 100


import quapy as qp

data = qp.datasets.fetch_twitter('wb', min_df=3, pickle=True, for_model_selection=False)

X, y = data.training.Xy

cls = LogisticRegression(C=0.0001, random_state=0)


posteriors = cross_val_predict(cls, X=X, y=y, method='predict_proba', n_jobs=-1, cv=3)

cls.fit(X, y)

Xte, yte = data.test.Xy

post_c1 = posteriors[y==0]
post_c2 = posteriors[y==1]
post_c3 = posteriors[y==2]


print(len(post_c1))
print(len(post_c2))
print(len(post_c3))

post_test = cls.predict_proba(Xte)

alpha = qp.functional.prevalence_from_labels(yte, classes=[0, 1, 2])


nbins = 20

plt.rcParams.update({'font.size': 7})

fig = plt.figure()
positions = np.asarray([2,1,0])
colors = ['r', 'g', 'b']

for i, post_set in enumerate([post_c1, post_c2, post_c3, post_test]):
    ax = fig.add_subplot(141+i, projection='3d')
    for post, c, z in zip(post_set.T, colors, positions):

        hist, bins = np.histogram(post, bins=nbins, density=True, range=[0,1])
        xs = (bins[:-1] + bins[1:])/2

        ax.bar(xs, hist, width=1/nbins, zs=z, zdir='y', color=c, ec=c, alpha=0.6)

    ax.yaxis.set_ticks(positions)
    ax.yaxis.set_ticklabels(['$y=1$', '$y=2$', '$y=3$'])
    ax.xaxis.set_ticks([])
    ax.xaxis.set_ticklabels([], minor=True)
    ax.zaxis.set_ticks([])
    ax.zaxis.set_ticklabels([], minor=True)


#plt.figure(figsize=(10,6))
#plt.show()
plt.savefig('./histograms.pdf')


