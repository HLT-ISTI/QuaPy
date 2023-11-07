import ternary
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.neighbors import KernelDensity
import plotly.figure_factory as ff

from data import LabelledCollection

scale = 100


# con ternary (una lib de matplotlib) salen bien pero no puedo crear contornos, o no se
# con plotly salen los contornos bien, pero es un poco un jaleo porque utiliza el navegador...

def plot_simplex_(ax, density, title='', fontsize=30, points=None):

    tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale)
    tax.heatmapf(density, boundary=True, style="triangular", colorbar=False, cmap='viridis') #cmap='magma')
    tax.boundary(linewidth=1.0)
    corner_fontsize = int(5*fontsize//6)
    tax.right_corner_label("$y=3$", fontsize=corner_fontsize)
    tax.top_corner_label("$y=2$", fontsize=corner_fontsize)
    tax.left_corner_label("$y=1$", fontsize=corner_fontsize)
    if title:
        tax.set_title(title, loc='center', y=-0.11, fontsize=fontsize)
    if points is not None:
        tax.scatter(points*scale, marker='o', color='w', alpha=0.25, zorder=10, s=5*scale)
    tax.get_axes().axis('off')
    tax.clear_matplotlib_ticks()

    return tax



from mpl_toolkits.axes_grid1 import make_axes_locatable
def plot_3class_problem(post_c1, post_c2, post_c3, post_test, alpha, bandwidth):
    post_c1 = np.flip(post_c1, axis=1)
    post_c2 = np.flip(post_c2, axis=1)
    post_c3 = np.flip(post_c3, axis=1)
    post_test = np.flip(post_test, axis=1)

    size_=10
    fig = ternary.plt.figure(figsize=(5*size_, 1*size_))
    fig.tight_layout()
    ax1 = fig.add_subplot(1, 4, 1)
    divider = make_axes_locatable(ax1)
    ax2 = fig.add_subplot(1, 4, 2)
    divider = make_axes_locatable(ax2)
    ax3 = fig.add_subplot(1, 4, 3)
    divider = make_axes_locatable(ax3)
    ax4 = fig.add_subplot(1, 4, 4)
    divider = make_axes_locatable(ax4)

    kde1 = KernelDensity(bandwidth=bandwidth).fit(post_c1)
    kde2 = KernelDensity(bandwidth=bandwidth).fit(post_c2)
    kde3 = KernelDensity(bandwidth=bandwidth).fit(post_c3)

    #post_c1 = np.concatenate([post_c1, np.eye(3, dtype=float)])
    #post_c2 = np.concatenate([post_c2, np.eye(3, dtype=float)])
    #post_c3 = np.concatenate([post_c3, np.eye(3, dtype=float)])

    #plot_simplex_(ax1, lambda x:0, title='$f_1(\mathbf{x})=p(s(\mathbf{x})|y=1)$')
    #plot_simplex_(ax2, lambda x:0, title='$f_1(\mathbf{x})=p(s(\mathbf{x})|y=1)$')
    #plot_simplex_(ax3, lambda x:0, title='$f_1(\mathbf{x})=p(s(\mathbf{x})|y=1)$')
    def density(kde):
        def d(p):
            return np.exp(kde([p])).item()
        return d

    plot_simplex_(ax1, density(kde1.score_samples), title='$p_1$')
    plot_simplex_(ax2, density(kde2.score_samples), title='$p_2$')
    plot_simplex_(ax3, density(kde3.score_samples), title='$p_3$')
    #plot_simplex(ax1, post_c1, np.exp(kde1.score_samples(post_c1)), title='$f_1(\mathbf{x})=p(s(\mathbf{x})|y=1)$') #, savepath='figure/y1.png')
    #plot_simplex(ax2, post_c2, np.exp(kde2.score_samples(post_c2)), title='$f_2(\mathbf{x})=p(s(\mathbf{x})|y=2)$') #, savepath='figure/y2.png')
    #plot_simplex(ax3, post_c3, np.exp(kde3.score_samples(post_c3)), title='$f_3(\mathbf{x})=p(s(\mathbf{x})|y=3)$') #, savepath='figure/y3.png')

    def mixture_(prevs, kdes):
        def m(p):
            total_density = 0
            for prev, kde in zip(prevs, kdes):
                log_density = kde.score_samples([p]).item()
                density = np.exp(log_density)
                density *= prev
                total_density += density
            #print(total_density)
            return total_density
        return m

    title = '$\mathbf{p}_{\mathbf{\\alpha}} = \sum_{i \in n} \\alpha_i p_i$'

    plot_simplex_(ax4, mixture_(alpha, [kde1, kde2, kde3]), title=title, points=post_test)

    #ternary.plt.show()
    ternary.plt.savefig('./simplex.pdf')


import quapy as qp


data = qp.datasets.fetch_twitter('wb', min_df=3, pickle=True, for_model_selection=False)

Xtr, ytr = data.training.Xy
Xte, yte = data.test.sampling(150, *[0.5, 0.1, 0.4]).Xy

cls = LogisticRegression(C=0.0001, random_state=0)

draw_from_training = False
if draw_from_training:
    post_tr = cross_val_predict(cls, Xtr, ytr, n_jobs=-1, method='predict_proba')
    post_c1 = post_tr[ytr==0]
    post_c2 = post_tr[ytr==1]
    post_c3 = post_tr[ytr==2]
    cls.fit(Xtr, ytr)
else:
    cls.fit(Xtr, ytr)
    post_te = cls.predict_proba(Xte)
    post_c1 = post_te[yte == 0]
    post_c2 = post_te[yte == 1]
    post_c3 = post_te[yte == 2]

post_test = cls.predict_proba(Xte)

alpha = qp.functional.prevalence_from_labels(yte, classes=[0, 1, 2])

print(f'test alpha {alpha}')
plot_3class_problem(post_c1, post_c2, post_c3, post_test, alpha, bandwidth=0.1)

