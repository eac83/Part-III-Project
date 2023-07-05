import numpy as np
from scipy import *
import camb
from camb import model, initialpower
from matplotlib import pyplot as plt

def linPowerSpec(h, Omega_m, Omega_b, ns, mnu, redshifts, As):
    pars = camb.CAMBparams()
    H0, ombh2, omch2, ns = h*100, Omega_b*h**2, (Omega_m-Omega_b)*h**2, ns
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu)
    pars.InitPower.set_params(ns=ns, As=As)
    pars.set_matter_power(redshifts=redshifts, kmax=1500)
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=10**(3.1), npoints=1001)

    return pars, results, kh, z, pk

def nonLinPowerSpec(h, Omega_m, Omega_b, ns, mnu, redshifts, As):
    pars2 = camb.CAMBparams()
    H0, ombh2, omch2, ns = h*100, Omega_b*h**2, (Omega_m-Omega_b)*h**2, ns
    pars2.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu)
    pars2.InitPower.set_params(ns=ns, As=As)
    pars2.set_matter_power(redshifts=redshifts, kmax=1500)
    pars2.NonLinear = model.NonLinear_both
    results2 = camb.get_results(pars2)
    results2.calc_power_spectra(pars2) 
    kh2, _, pk2 = results2.get_matter_power_spectrum(minkh=1e-4, maxkh=1000, npoints=200)
    return kh2, pk2

#FABLE Header Parameters
h = 0.679
Omega_m = 0.3065
redshifts = [2]

#other parameters defined as in Henden et al. 2018
Omega_b = 0.0483
ns = 0.9681
sigma8 = 0.8154

AsTrial = 2.018735662e-9  #starting value from a previous example
mnu = 0.0                 #probably

#calculate the true value of As
pars, results, kh, z, pk = linPowerSpec(h, Omega_m, Omega_b, ns, mnu, [0], AsTrial)
sigma8trial = np.array(results.get_sigma8())[0]
As = AsTrial * (sigma8/sigma8trial)**2     #A_s\propto\sigma_8^2

#calculate linear power spectrum
pars, results, kh, z, pk = linPowerSpec(h, Omega_m, Omega_b, ns, mnu, redshifts, As)

with open('./Box_Lin_PS_z2.csv', 'w') as f:
	f.write('k,P(k)\n')
	np.savetxt(f, np.c_[kh, pk[0].T], delimiter=',')

#calculate non-linear power spectrum
khNonLin, pkNonLin = nonLinPowerSpec(h, Omega_m, Omega_b, ns, mnu, redshifts, As)

with open('./Box_Non-Lin_PS_z2.csv', 'w') as f:
	f.write('k,P(k)\n')
	np.savetxt(f, np.c_[khNonLin, pkNonLin[0].T], delimiter=',')

fg, ax = plt.subplots()
ax.loglog()
ax.plot(kh, pk.T, linewidth=2, label=f'Linear, z={z}')
ax.plot(khNonLin, pkNonLin.T, linewidth=2, label=f'Non-Linear, z={z}')
ax.set_xlabel(r'$k/[(\mathrm{Mpc}/h)^{-1}]$')
ax.set_ylabel(r'$P(k)[(\mathrm{Mpc}/h)^3$')
ax.set_title(f'Total Matter Power Spectrum at $z={redshifts[0]}$')
ax.legend()
print('Plotted!')
fg.show()
input()
