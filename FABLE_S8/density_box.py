'''Calculate density profiles of haloes in box'''

# pylint: disable=unbalanced-tuple-unpacking

import sys
import time
import numpy as np
import h5py
import functions as fn
import box
import halo

#read in arguments
_, z, log_m_min, log_m_max, IS_VERBOSE = tuple(sys.argv)
z, log_m_min, log_m_max = float(z), float(log_m_min), float(log_m_max)
if is_verbose:
    print(f'z = {z}')
    print(f'{log_m_min} < log(m) < {log_m_max}')

#consts
x_BINS = np.logspace(-2, 0, 21)
m_BINS = np.logspace(log_m_min, log_m_max)
NO_BATCH = 100
OUTFILE = fn.HERE_PATH/'Density_Profiles'/f'''box_{z}_{np.log10(m_bins[0])}
_{np.log10(m_bins[-1])}.csv'''

#create box and halo
FOF_PATH = fn.BOX_PATH/f'fof_subhalo_tab_{fn.SHOT[z]}.hdf5'

box_f = box.Box(fn.BOX_PATH, fn.SHOT[z])

with h5py.File(fof_path) as fof:
    mass = np.array(fof.get('Group/GroupMass'))*1e10
    ndx = np.where((mass >= m_BINS[0]) & (mass <= m_BINS[-1]))
    halo_pos = np.array(fof.get('Group/GroupPos'))[ndx]
    halo_r200c = np.array(fof.get('Group/Group_R_Crit200'))[ndx]

NO_HALOS = halo_r200c.shape[0]
if is_verbose:
    print(f'Found {NO_HALOS} halos.')

halos = [halo.Halo(halo_pos[i, :], r200c) for i, r200c
         in enumerate(halo_r200c)]

#get density profiles
with open(OUTFILE, 'w') as f:
    header = 'i, M,'
    for i, _ in enumerate(x_BINS[:-1]):
        header += f'rho_{i},'
    header += 'r200\n'
    f.write(header)

if IS_VERBOSE:
    print('Calculating density profiles')
    START = time.perf_counter()

for i in range(0, NO_HALOS-(NO_HALOS%NO_BATCH), NO_BATCH):
    if IS_VERBOSE:
        begin = time.perf_counter()
    [halo_i.get_density(box_f, x_BINS) for halo_i in halos[i:i+NO_BATCH]]
    if IS_VERBOSE:
        end = time.perf_counter()
        print('------------------------')
        print(f'{i+NO_BATCH} calculated')
        print(f'Time for {NO_BATCH}: {begin-end} s')
        print(f'Time for {i+NO_BATCH}: {START-end} s')

    #save to file
    with open(OUTFILE, 'a') as f:
        for j in range(i, i+NO_BATCH):
            np.savetxt(f, np.c_[j, halos[j].mass,
                                halos[j].density[:, np.newaxis].T,
                                halos[j].r200c], delimiter=',')

if IS_VERBOSE:
    begin = time.perf_counter()

[halo_i.get_density(box_f, x_BINS) for halo_i
 in halos[NO_HALOS-(NO_HALOS%NO_BATCH):]]

if IS_VERBOSE:
    end = time.perf_counter()
    print('------------------------')
    print(f'{NO_HALOS} calculated')
    print(f'Time for {NO_HALOS-(NO_HALOS%NO_BATCH)}: {begin-end} s')
    print(f'Time for {NO_HALOS}: {START-end} s')

    with open(OUTFILE, 'a') as f:
        for j in range(NO_HALOS-(NO_HALOS%NO_BATCH), NO_HALOS):
            np.savetxt(f, np.c_[j, halos[j].mass,
                                halos[j].density[:, np.newaxis].T,
                                halos[j].r200c], delimiter=',')

if IS_VERBOSE:
    print('All saved to file')
