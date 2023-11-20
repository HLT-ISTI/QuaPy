import numpy as np
from scipy.stats import norm, uniform, expon
from scipy.special import rel_entr


def kld_approx(p, q, resolution=100):
    steps = np.linspace(-5, 5, resolution)
    integral = 0
    for step_i1, step_i2 in zip(steps[:-1], steps[1:]):
        base = step_i2-step_i1
        middle = (step_i1+step_i2)/2
        pi = p.pdf(middle)
        qi = q.pdf(middle)
        integral += (base * pi*np.log(pi/qi))
    return integral


def integrate(f, resolution=10000):
    steps = np.linspace(-10, 10, resolution)
    integral = 0
    for step_i1, step_i2 in zip(steps[:-1], steps[1:]):
        base = step_i2 - step_i1
        middle = (step_i1 + step_i2) / 2
        integral += (base * f(middle))
    return integral


def kl_analytic(m1, s1, m2, s2):
    return np.log(s2/s1)+ (s1*s1 + (m1-m2)**2) / (2*s2*s2) - 0.5


def montecarlo(p, q, trials=1000000):
    xs = p.rvs(trials)
    ps = p.pdf(xs)
    qs = q.pdf(xs)
    return np.mean(np.log(ps/qs))


def montecarlo2(p, q, trials=100000):
    #r = norm(-3, scale=5)
    r = uniform(-10, 20)
    # r = expon(loc=-6)
    #r = p
    xs = r.rvs(trials)
    rs = r.pdf(xs)
    print(rs)
    ps = p.pdf(xs)+0.0000001
    qs = q.pdf(xs)+0.0000001
    return np.mean((ps/rs)*np.log(ps/qs))


def wrong(p, q, trials=100000):
    xs = np.random.uniform(-10,10, trials)
    ps = p.pdf(xs)
    qs = q.pdf(xs)
    return np.mean(ps*np.log(ps/qs))


p = norm(loc=0, scale=1)
q = norm(loc=1, scale=1)


integral_approx = kld_approx(p, q)
analytic_solution = kl_analytic(p.kwds['loc'], p.kwds['scale'], q.kwds['loc'], q.kwds['scale'])
montecarlo_approx = montecarlo(p, q)
montecarlo_approx2 = montecarlo2(p, q)
wrong_approx = wrong(p, q)

print(f'{analytic_solution=:.10f}')
print()
print(f'{integral_approx=:.10f}')
print(f'integra error = {np.abs(analytic_solution-integral_approx):.10f}')
print()
print(f'{montecarlo_approx=:.10f}')
print(f'montecarlo error = {np.abs(analytic_solution-montecarlo_approx):.10f}')
print()
print(f'{montecarlo_approx2=:.10f}')
print(f'montecarlo2 error = {np.abs(analytic_solution-montecarlo_approx2):.10f}')
print()
print(f'{wrong_approx=:.10f}')
print(f'wrong error = {np.abs(analytic_solution-wrong_approx):.10f}')

