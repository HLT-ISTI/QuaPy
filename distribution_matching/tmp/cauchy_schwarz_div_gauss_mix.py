import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import norm, multivariate_normal, Covariance
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import KDTree

# def CauchySchwarzDivGaussMix(pi_i, mu_i, Lambda_i, tau_i, nu_i, Omega_i):
#     Lambda_i_inv =


# def z()


nD=2
mu = [[0.1,0.1], [0.2, 0.3], [1, 1]]      # centers
bandwidth = 0.10   # bandwidth and scale (standard deviation)
standard_deviation = bandwidth
variance = standard_deviation**2

mus = np.asarray(mu).reshape(-1,nD)
kde = KernelDensity(bandwidth=bandwidth).fit(mus)

x = [0.5,0.2]

print('with scikit-learn KDE')
prob = np.exp(kde.score_samples(np.asarray(x).reshape(-1,nD)))
print(prob)

# univariate
# N = norm(loc=mu, scale=scale)
# prob = N.pdf(x)
# print(prob)

# multivariate

print('with scipy multivariate normal')
npoints = mus.shape[0]
probs = sum(multivariate_normal(mean=mu_i, cov=variance).pdf(x) for mu_i in mu)
probs /= npoints
print(probs)

print('with scikit learn rbf_kernel')
x = np.asarray(x).reshape(-1,nD)
gamma = 1/(2*variance)
gram = rbf_kernel(mus, x, gamma=gamma)
const = 1/np.sqrt(((2*np.pi)**nD) * (variance**(nD)))
gram *= const
print(gram)
print(np.sum(gram)/npoints)

print('with stackoverflow answer')
from scipy.spatial.distance import pdist, cdist, squareform
import scipy
  # this is an NxD matrix, where N is number of items and D its dimensionalites
pairwise_sq_dists = cdist(mus, x, 'euclidean')
print(pairwise_sq_dists)
K = np.exp(-pairwise_sq_dists / variance)
print(K)
print(np.sum(K)/npoints)

print("trying with scipy multivariate on more than one instance")
probs = sum(multivariate_normal(mean=x.reshape(-nD), cov=variance).pdf(mus))
probs /= npoints
print(probs)

import sys
sys.exit(0)

# N1 = multivariate_normal(mean=mu[0], cov=variance)
# prob1 = N1.pdf(x)
# N2 = multivariate_normal(mean=mu[1], cov=variance)
# prob2 = N2.pdf(x)
# print(prob1+prob2)

cov = Covariance.from_diagonal([variance, variance])
print(cov.covariance)
precision_matrix = np.asarray([[1/variance, 0],[0, 1/variance]])
print(Covariance.from_precision(precision_matrix).covariance)
print(np.linalg.inv(precision_matrix))
print(np.linalg.inv(cov.covariance))

print('-'*100)

nD=2
mu = np.asarray([[0.1,0.5]])      # centers
bandwidth = 0.10   # bandwidth and scale (standard deviation)
standard_deviation = bandwidth
variance = standard_deviation**2

mus = np.asarray([mu]).reshape(-1,nD)
kde = KernelDensity(bandwidth=bandwidth).fit(mus)

x = np.asarray([0.5,0.2])

prob = np.exp(kde.score_samples(np.asarray(x).reshape(-1,nD)))
print(prob)

probs = sum(multivariate_normal(mean=mu_i, cov=variance).pdf(x) for mu_i in mu) / len(mu)
probs = sum(multivariate_normal(mean=[0,0], cov=1).pdf((x-mu_i)/bandwidth) for mu_i in mu) / len(mu)
probs = sum(multivariate_normal(mean=[0,0], cov=variance).pdf(x-mu_i) for mu_i in mu) / len(mu)
print(probs)
# N1 = multivariate_normal(mean=mu[0], cov=variance)
# prob1 = N1.pdf(x)
# N2 = multivariate_normal(mean=mu[1], cov=variance)
# prob2 = N2.pdf(x)
# print(prob1+prob2)

h=0.1
D=4

print(np.sqrt((4**2) * (5**2)))
print(4*5)

