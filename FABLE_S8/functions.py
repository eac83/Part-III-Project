"""Store functions and constants used throughout project"""

from pathlib import Path
import numpy as np
import h5py
from astropy import constants as const
import halo

#apply periodic boundary conditions
def nearest(coords, box_size):
    """Take in a displacement and box side length, return shortest
    distance given periodic boundary conditions."""
    for axis in range(3):
        ndx = np.where((coords[:, axis] > box_size/2))
        coords[ndx, axis] = box_size - coords[ndx, axis]
        ndx = np.where((coords[:, axis] < -box_size/2))
        coords[ndx, axis] = box_size + coords[ndx, axis]
    return coords

#calculate mass averaged density profiles
def get_mass_avg_density(halos, m_bins):
    """Take in an array of Halo objects and edges of mass bins, return the
    mass averaged density profiles in the mass bins given."""
    num_halos = np.array([])
    halo_masses = np.array([halo.mass*1e10 for halo in halos])
    halo_density = np.array([halo.density for halo in halos]).T
    halo_r200c = np.array([halo.r200c for halo in halos])
    avg_halos = np.array([])
    for i, m_i in enumerate(m_bins[:-1]):
        ndx = np.array(np.where((halo_masses >= m_i)&(halo_masses <
                                                      m_bins[i+1])),
                       dtype=int)[0]
        num_halos = np.append(num_halos, ndx.shape[0])
        if ndx.size > 0:
            density_avg = (np.sum(halo_masses[ndx]*halo_density[:, ndx],
                                  axis=1) / np.sum(halo_masses[ndx]))
            r200c_avg = (np.sum(halo_masses[ndx])
                         / np.sum(halo_masses[ndx]/halo_r200c[ndx]))
            avg_halos = np.append(avg_halos, halo.Halo(np.array([0, 0, 0]),
                                                       r200c_avg))
            avg_halos[-1].mass = np.sqrt(m_i*m_bins[i+1])
            avg_halos[-1].density = density_avg
    return avg_halos, num_halos

def get_halo_mass_fn(z, m_bins):
    """Take in a redshift and edges of mass bins, return the arithmetic mean of
    the bins and the halo mass function of the FABLE box at that redshift using
    those bins."""
    fof_path = BOX_PATH/f'fof_subhalo_tab_{SHOT[z]}.hdf5'
    with h5py.File(fof_path) as fof:
        size = fof['Header'].attrs['BoxSize']/1e3
        mass = np.array(fof.get('Group/Group_M_Crit200'))*1e10
        hist = np.array(np.histogram(mass, bins=m_bins))
        d_b = np.diff(np.log10(m_bins))*size**3
        m_mean = np.array([])
        for i in range(m_bins.shape[0]-1):
            idx = np.where((mass >= m_bins[i])&(mass < m_bins[i+1]))
            m_mean = np.append(m_mean, np.mean(mass[idx]))
            if np.isnan(m_mean[i]):
                m_mean[i] = 0
    return m_mean, hist[0]/d_b

def get_power(box_f, halos, k, m_bins, z, no_halos):
    """Returns the 1h power spectrum given a Box object and a list of Halo
    objects."""
    log_m = np.log10(np.sqrt(m_bins[:-1]*m_bins[1:]))
    H = 100*box_f.h*1e3
    rho_crit = 3*H**2/(8*np.pi*const.G)
    rho_crit = (rho_crit * (const.pc*1e6) / (const.M_sun) / box_f.h**2).value
    rho_bar = rho_crit * box_f.omega_m

    m_mean, n = get_halo_mass_fn(z, m_bins)
    m_mean = m_mean[np.where(no_halos > 0)]
    n = n[np.where(no_halos > 0)]
    p1h = np.array([])
    u = np.array([halo.u for halo in halos])
    for i, _ in enumerate(k):
        p1h = np.append(p1h, np.trapz(n*(m_mean/rho_bar)**2*np.abs(u[:, i])**2,
                                      log_m[np.where(no_halos > 0)]))
    return p1h

#convert redshift to snapshot number
SHOT = {2:'015', 0:'025'}
ZHOT = {3:'039', 2:'059', 1:'079', 0:'099'}
XHOT = {2:'161', 0:'686'}

#often used paths
HERE_PATH = Path('/data/ERCblackholes4/eac83/FABLE_S8')
BOX_PATH = Path('/data/ERCblackholes4/FABLE/boxes/L40_512_MDRIntRadioEff')
ZOOMS_PATH = Path('/data/ERCblackholes4/FABLE/zooms/MDRInt')
MODELS_PATH = Path('/data/ERCblackholes4/eac83')

#list of models and their dictionaries
MODELS = ['REF', 'ModM2E-4', 'ModM2E-4P0.2', 'ModM2E-4P5', 'ModM2E-4P10',
          'ModM7E-4']
COLORS = {'REF':'k', 'ModM2E-4':'b', 'ModM2E-4P0.2':'r', 'ModM2E-4P5':'g',
          'ModM2E-4P10':'orange', 'ModM7E-4':'purple'}
LABELS = {'REF':'Reference', 'ModM2E-4':r'$M_\mathrm{thresh}: \, 2, \, P:\,1$',
          'ModM2E-4P0.2':r'$M_\mathrm{thresh}: \, 2, \, P: \, 0.2$',
          'ModM2E-4P5':r'$M_\mathrm{thresh}: \, 2, \, P: \,5$',
          'ModM2E-4P10':r'$M_\mathrm{thresh}: \, 2, \, P: \,10$',
          'ModM7E-4':r'$M_\mathrm{thresh}: \, 7,\, P: \, 1$'}

#often used plotting strings
LEGEND_TITLE = (r'$M_\mathrm{thresh} [10^6 h^{-1}'
                + r'\mathrm{M}_\odot],\,P/P_\mathrm{fid}$')
