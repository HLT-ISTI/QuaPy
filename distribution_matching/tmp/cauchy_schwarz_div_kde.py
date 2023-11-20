import numpy as np
from scipy.stats import multivariate_normal
from scipy import optimize


def cauchy_schwarz_divergence_kde(L:list, Xte:np.ndarray, bandwidth:float, alpha:np.ndarray):
    """

    :param L: a list of np.ndarray (instances x dimensions) with the Li being the instances of class i
    :param Xte: an np.ndarray (instances x dimensions)
    :param bandwidth: the bandwidth of the kernel
    :param alpha: the mixture parameter
    :return: the Cauchy-Schwarz divergence between the validation KDE mixture distribution (with mixture paramerter
        alpha) and the test KDE distribution
    """

    n = len(L) # number of classes
    K, D = Xte.shape # number of test instances, and number of dimensions
    Kinv = 1/K

    # the lengths of each block
    l = np.asarray([len(Li) for Li in L])

    # contains the a_i / l_i
    alpha_r = alpha / l
    alpha2_r_sum = np.sum(alpha * alpha_r)  # contains the sum_i a_i**2 / l_i

    h = bandwidth

    # we will only use the bandwidth (h) between two gaussians with covariance matrix a "scalar matrix" h**2
    cov_mix_scalar = 2*h*h   # corresponds to a bandwidth of sqrt(2)*h

    # constant
    C = ((2*np.pi)**(-D/2))*h**(-D)

    Kernel = multivariate_normal(mean=np.zeros(D), cov=cov_mix_scalar)
    K0 = Kernel.pdf(np.zeros(D))


    def compute_block_E():
        kernel_block_E = []
        for i,Li in enumerate(L):
            acc = 0
            for x_ji in Li: #optimize...
                for x_k in Xte: #optimize...
                    acc += Kernel.pdf(x_ji - x_k) #optimize...
            kernel_block_E.append(acc)
        return np.asarray(kernel_block_E)


    def compute_block_F_hash():
        # this can be computed entirely at training time
        Khash = {}
        for a in range(n):
            for b in range(l[a]):
                for i in range(n):
                    for j in range(l[i]): # this for, and index j, can be supressed and store the sum across j
                        Khash[(a,b,i,j)] = Kernel.pdf(L[i][j]-L[a][b])
        return Khash


    def compute_block_Ktest():
        # this can be optimized in several ways, starting by computing only the lower diagonal triangle... and remove
        # then the K0 which is not needed after that
        acc = 0
        for x_i in Xte:
            for x_j in Xte:
                acc += Kernel.pdf(x_i-x_j)
        return acc


    def compute_block_F():
        F = 0
        for a in range(n):
            tmp_b = 0
            for b in range(l[a]):
                tmp_i = 0
                for i in range(n):
                    tmp_j = 0
                    for j in range(l[i]):
                        tmp_j += Fh[(a, b, i, j)]
                    tmp_i += (alpha_r[i] * tmp_j)
                tmp_b += tmp_i
            F += (alpha_r[a] * tmp_b)
        return F


    E = compute_block_E()
    Fh = compute_block_F_hash()
    # Ktest = compute_block_Ktest()
    F = compute_block_F()

    C1 = K*Kinv*Kinv*C
    C2 = 2 * np.sum([Kernel.pdf(Xte[k]-Xte[k_p]) for k in range(K) for k_p in range(k)])

    partA = -np.log(Kinv * (alpha_r @ E))
    partB = 0.5*np.log(C*alpha2_r_sum + F - (K0*alpha2_r_sum))
    # partC = 0.5*np.log(Kinv) + 0.5*np.log(C + Kinv*Ktest - K0)
    partC = 0.5*np.log(C1+C2)

    Dcs = partA + partB + partC

    return Dcs


L = [
    np.asarray([
        [-1,-1,-1]
    ]),
    np.asarray([
        [0,0,0],
    ]),
    np.asarray([
        [0,0,0.1],
        [1,1,1],
        [3,3,1],
    ]),
    np.asarray([
        [1,0,0]
    ]),
    np.asarray([
        [0,1,0]
    ])
]
Xte = np.asarray(
    [[0,0,0],
     [0,0,0],
     [1,0,0],
     [0,1,0]]
)
bandwidth=0.01
alpha=np.asarray([0, 2/4, 0, 1/4, 1/4])

div = cauchy_schwarz_divergence_kde(L, Xte, bandwidth, alpha)
print(div)

def divergence(alpha):
    return cauchy_schwarz_divergence_kde(L, Xte, bandwidth, alpha)


# the initial point is set as the uniform distribution
n_classes = len(L)
uniform_distribution = np.full(fill_value=1 / n_classes, shape=(n_classes,))

# solutions are bounded to those contained in the unit-simplex
bounds = tuple((0, 1) for _ in range(n_classes))  # values in [0,1]
constraints = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})  # values summing up to 1
#print('searching for alpha')
r = optimize.minimize(divergence, x0=uniform_distribution, method='SLSQP', bounds=bounds, constraints=constraints)
sol = r.x
for x in sol:
    print(f'{x:.4f}')
print(cauchy_schwarz_divergence_kde(L, Xte, bandwidth, sol))