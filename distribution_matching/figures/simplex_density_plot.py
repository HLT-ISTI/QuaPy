import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity

from data import LabelledCollection

scale = 200


# con ternary (una lib de matplotlib) salen bien pero no puedo crear contornos, o no se
# con plotly salen los contornos bien, pero es un poco un jaleo porque utiliza el navegador...

def plot_simplex_(ax, density, title='', fontsize=9, points=None):
    import ternary

    tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale)
    tax.heatmapf(density, boundary=True, style="triangular", colorbar=False, cmap='viridis') #cmap='magma')
    tax.boundary(linewidth=1.0)
    corner_fontsize = 5*fontsize//6
    tax.right_corner_label("$y=3$", fontsize=corner_fontsize)
    tax.top_corner_label("$y=2$", fontsize=corner_fontsize)
    tax.left_corner_label("$y=1$", fontsize=corner_fontsize)
    if title:
        tax.set_title(title, loc='center', y=-0.11, fontsize=fontsize)
    if points is not None:
        tax.scatter(points*scale, marker='o', color='w', alpha=0.25, zorder=10)
    tax.get_axes().axis('off')
    tax.clear_matplotlib_ticks()

    return tax


def plot_simplex(ax, coord, kde_scores, title='', fontsize=11, points=None, savepath=None):
    import plotly.figure_factory as ff

    tax = ff.create_ternary_contour(coord.T, kde_scores, pole_labels=['y=1', 'y=2', 'y=3'],
                                interp_mode='cartesian',
                                ncontours=20,
                                colorscale='Viridis',
                                showscale=True,
                                title=title)
    if savepath is None:
        tax.show()
    else:
        tax.write_image(savepath)
    return tax

from mpl_toolkits.axes_grid1 import make_axes_locatable
def plot_3class_problem(post_c1, post_c2, post_c3, post_test, alpha, bandwidth):
    import ternary

    post_c1 = np.flip(post_c1, axis=1)
    post_c2 = np.flip(post_c2, axis=1)
    post_c3 = np.flip(post_c3, axis=1)
    post_test = np.flip(post_test, axis=1)

    fig = ternary.plt.figure(figsize=(26, 3))
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

    plot_simplex_(ax1, density(kde1.score_samples), title='$f_1(\mathbf{x})=p(s(\mathbf{x})|y=1)$')
    plot_simplex_(ax2, density(kde2.score_samples), title='$f_2(\mathbf{x})=p(s(\mathbf{x})|y=2)$')
    plot_simplex_(ax3, density(kde3.score_samples), title='$f_3(\mathbf{x})=p(s(\mathbf{x})|y=3)$')
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

    title = '$\sum_{i \in \mathcal{Y}} \\alpha_i f_i(\mathbf{x})$'

    plot_simplex_(ax4, mixture_(alpha, [kde1, kde2, kde3]), title=title, points=post_test)
    #mixture(alpha, [kde1, kde2, kde3])

    #post_test = np.concatenate([post_test, np.eye(3, dtype=float)])
    #test_scores = sum(alphai*np.exp(kdei.score_samples(post_test)) for alphai, kdei in zip(alpha, [kde1,kde2,kde3]))
    #plot_simplex(ax4, post_test, test_scores, title=title, points=post_test)

    ternary.plt.show()


import quapy as qp


data = qp.datasets.fetch_twitter('wb', min_df=3, pickle=True, for_model_selection=False)

X, y = data.training.Xy

cls = LogisticRegression(C=0.0001, random_state=0)

Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=0.7, stratify=y, random_state=0)

cls.fit(Xtr, ytr)

test = LabelledCollection(Xte, yte)
test = test.sampling(100, *[0.2, 0.1, 0.7])

Xte, yte = test.Xy

post_c1 = cls.predict_proba(Xte[yte==0])
post_c2 = cls.predict_proba(Xte[yte==1])
post_c3 = cls.predict_proba(Xte[yte==2])

post_test = cls.predict_proba(Xte)
print(post_test)
alpha = qp.functional.prevalence_from_labels(yte, classes=[0, 1, 2])

#post_c1 = np.random.dirichlet([10,3,1], 30)
#post_c2 = np.random.dirichlet([1,11,6], 30)
#post_c3 = np.random.dirichlet([1,5,20], 30)
#post_test = np.random.dirichlet([5,1,6], 100)
#alpha = [0.5, 0.3, 0.2]


print(f'test alpha {alpha}')
plot_3class_problem(post_c1, post_c2, post_c3, post_test, alpha, bandwidth=0.1)

