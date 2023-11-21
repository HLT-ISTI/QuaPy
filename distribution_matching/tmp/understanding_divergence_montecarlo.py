import numpy as np
from scipy.stats import norm


EPS=1e-8
LARGE_NUM=1000000
TRIALS=LARGE_NUM

# calculamos: D(p||q)
# se asume:
# q = distribución de referencia (en nuestro caso: el mixture de KDEs en training)
# p = distribución (en nuestro caso: KDE del test)
# N = TRIALS el número de trials para montecarlo
# n = numero de clases, en este ejemplo n=3


# ejemplo de f-generator function: Squared Hellinger Distance
def hd2(u):
    v = (np.sqrt(u)-1)
    return v*v


# esta funcion estima la divergencia entre dos mixtures (univariates)
def integrate(p, q, f=hd2, resolution=TRIALS, epsilon=EPS):
    xs = np.linspace(-50, 50, resolution)
    ps = p.pdf(xs)+epsilon
    qs = q.pdf(xs)+epsilon
    rs = ps/qs
    base = xs[1]-xs[0]
    return np.sum(base * f(rs) * qs)


# esta es la aproximación de montecarlo según está en el artículo
# es decir, sampleando la q, y haciendo la media (1/N). En el artículo
# en realidad se "presamplean" N por cada clase y luego se eligen los que
# corresponda a cada clase. Aquí directamente sampleamos solo los
# que correspondan.
def montecarlo(p, q, f=hd2, trials=TRIALS, epsilon=EPS):
    xs = q.rvs(trials)
    ps = p.pdf(xs)+epsilon
    qs = q.pdf(xs)+epsilon
    N = trials
    return (1/N)*np.sum(f(ps/qs))


# esta es la aproximación de montecarlo asumiendo que vayamos con todos los N*n
# puntos (es decir, si hubieramos sampleado de una "q" en la que todas las clases
# eran igualmente probables, 1/3) y luego reescalamos los pesos con "importance
# weighting" <-- esto no está en el artículo, pero me gusta más que lo que hay
# porque desaparece el paso de "seleccionar los que corresponden por clase"
def montecarlo_importancesampling(p, q, r, f=hd2, trials=TRIALS, epsilon=EPS):
    xs = r.rvs(trials)  # <- distribución de referencia (aquí: class weight uniforme)
    rs = r.pdf(xs)
    ps = p.pdf(xs)+epsilon
    qs = q.pdf(xs)+epsilon
    N = trials
    importance_weight = qs/rs
    return (1/N)*np.sum(f(ps/qs)*(importance_weight))


# he intentado implementar la variante que propne Juanjo pero creo que no la he
# entendido bien. Tal vez haya que reescalar el peso por 1/3 o algo así, pero no
# doy con la tecla...
def montecarlo_classweight(p, q, f=hd2, trials=TRIALS, epsilon=EPS):
    xs_1 = q.rvs_fromclass(0, trials)
    xs_2 = q.rvs_fromclass(1, trials)
    xs_3 = q.rvs_fromclass(2, trials)
    xs = np.concatenate([xs_1, xs_2, xs_3])
    weights = np.asarray([[alpha_i]*trials for alpha_i in q.alphas]).flatten()
    ps = p.pdf(xs)+epsilon
    qs = q.pdf(xs)+epsilon
    N = trials
    n = q.n
    return (1/(N))*np.sum(weights*f(ps/qs))


class Q:
    """
    Esta clase es un mixture de gausianas.

    :param locs: medias, una por clase
    :param scales: stds, una por clase
    :param alphas: peso, uno por clase
    """
    def __init__(self, locs, scales, alphas):
        self.qs = []
        for loc, scale in zip(locs, scales):
            self.qs.append(norm(loc=loc, scale=scale))
        assert np.isclose(np.sum(alphas), 1), 'alphas do not sum up to 1!'
        self.alphas = np.asarray(alphas) / np.sum(alphas)

    def pdf(self, xs):
        q_xs = np.vstack([
            q_i.pdf(xs) * alpha_i for q_i, alpha_i in zip(self.qs, self.alphas)
        ])
        v = q_xs.sum(axis=0)
        if len(v)==1:
            v = v[0]
        return v

    @property
    def n(self):
        return len(self.alphas)

    def rvs_fromclass(self, inclass, trials):
        return self.qs[inclass].rvs(trials)

    def rvs(self, trials):
        variates = []
        added = 0
        for i, (q_i, alpha_i) in enumerate(zip(self.qs, self.alphas)):
            trials_i = int(trials*alpha_i) if (i < self.n-1) else trials-added
            variates_i = q_i.rvs(trials_i)
            variates.append(variates_i)
            added += len(variates_i)

        return np.concatenate(variates)

    @property
    def locs(self):
        return np.asarray([x.kwds['loc'] for x in self.qs])

    @property
    def scales(self):
        return np.asarray([x.kwds['scale'] for x in self.qs])

    @classmethod
    def change_priors(cls, q, alphas):
        assert len(alphas)==len(q.alphas)
        return Q(locs=q.locs, scales=q.scales, alphas=alphas)

# distribucion de test
p = Q(locs=[1, 1.5], scales=[1,1], alphas=[0.8, 0.2])

# distribucion de training (mixture por clase)
q = Q(locs=[0, 0.5, 1.2], scales=[1,1,1.5], alphas=[0.1, 0.2, 0.7])

# distribucion de referencia (mixture copia de q, con pesos de clase uniformes)
r = Q.change_priors(q, alphas=[1/3, 1/3, 1/3])


integral_approx = integrate(p, q)
montecarlo_approx = montecarlo(p, q)
montecarlo_approx2 = montecarlo_importancesampling(p, q, r)
montecarlo_approx3 = montecarlo_classweight(p, q)

print(f'{integral_approx=:.10f}')
print()
print(f'{montecarlo_approx=:.10f}, error={np.abs(integral_approx-montecarlo_approx):.10f}')
print()
print(f'{montecarlo_approx2=:.10f}, error={np.abs(integral_approx-montecarlo_approx2):.10f}')
print()
print(f'{montecarlo_approx3=:.10f}, error={np.abs(integral_approx-montecarlo_approx3):.10f}')


