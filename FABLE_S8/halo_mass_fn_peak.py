'''plot m^2n(m) against m at z=2, z=0'''
import numpy as np
from matplotlib import pyplot as plt
import functions as fn

M_BINS_0 = np.logspace(10.5, 14, 8)
M_BINS_2 = np.logspace(10.5, 13.2, 8)

M_0 = np.sqrt(M_BINS_0[:-1]*M_BINS_0[1:])
M_2 = np.sqrt(M_BINS_2[:-1]*M_BINS_2[1:])

_, n_0 = fn.get_halo_mass_fn(0, M_BINS_0)
_, n_2 = fn.get_halo_mass_fn(0, M_BINS_2)

plt.loglog(M_2, M_2*n_2, lw=2, label='2')
plt.loglog(M_0, M_0*n_0, lw=2, label='0')

plt.xlabel(r'$z$')
plt.ylabel(r'$M \mathrm{d}N/\mathrm{d}\log M [10^{10} h^{-1} \mathrm{M}_\odot]$')
plt.legend(title=r'$z$')
plt.show()
