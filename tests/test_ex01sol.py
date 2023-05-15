import matplotlib.pyplot as plt
import numpy as np


mu = 0.000001
k = 1/mu



print(np.finfo(float))

def u_gen(x):
    return np.expm1(k * x) / np.expm1(k)


def u_spec(x):

    cut_low = 0.002
    cut_high = 10

    values = np.full(x.shape[-1], np.nan)

    values[k*x < cut_low] = 0.0

    inds = k*x >= cut_low

    la = np.zeros_like(values[inds])
    lb = np.zeros_like(values[inds])

    inds2 = k*x[inds] > cut_high
    inds3 = k*x[inds] <= cut_high
    la[inds2] = k*x[inds][inds2]
    la[inds3] = np.log(np.expm1(k*x[inds][inds3]))

    if k > cut_high:
        lb = k
    else:
        lb = np.log(np.expm1(k))

    values[inds] = np.exp(la - lb)

    return values

print(u_spec(np.linspace(0, 1, 101)))


xx = np.linspace(0.9999, 1, 101)

plt.figure()

plt.plot(xx, u_gen(xx))
# plt.plot(xx, [u_spec(x) for x in xx])
plt.plot(xx, u_spec(xx))


plt.show()




