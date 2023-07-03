import numpy as np
import h5py
import pandas as pd
import time
from pathlib import Path
import functions as fn
import Box
import Halo

#variables
z = 2
x = np.logspace(-2, 0, 21)
m = np.logspace(10.5, 13.2)
verbose = True
N = 100

#consts
outfile = fn.here_path/'Density_Profiles'/f'box_{z}.csv'

#create box and halo
fof_path = fn.box_path/f'fof_subhalo_tab_{fn.shot[z]}.hdf5'

box = Box.Box(fn.box_path, fn.shot[z])

with h5py.File(fof_path) as fof:
	mass = np.array(fof.get('Group/GroupMass'))*1e10
	ndx = np.where((mass>=m[0])&(mass<=m[-1]))
	halo_pos = np.array(fof.get('Group/GroupPos'))[ndx]
	halo_r200c = np.array(fof.get('Group/Group_R_Crit200'))[ndx]

no_halos = halo_r200c.shape[0]
if verbose:
	print(f'Found {no_halos} halos.')

halos = [Halo.Halo(halo_pos[i,:], r200c) for i, r200c in enumerate(halo_r200c)]

#get density profiles
with open(outfile, 'w') as f:
	header = 'i, M,'
	for i, _ in enumerate(x[:-1]):
		header += f'rho_{i},'
	header += 'r200\n'
	f.write(header)
	
if verbose:
	print('Calculating density profiles')
	start = time.perf_counter()

for i in range(0, no_halos-(no_halos%N), N):
	if verbose:
		begin = time.perf_counter()
	[halo.get_density(box, x) for halo in halos[i:i+N]]
	if verbose:
		end = time.perf_counter()
		print('------------------------')
		print(f'{i+N} calculated')
		print(f'Time for {N}: {end-begin} s')
		print(f'Time for {i+N}: {end-start} s')


	#save to file
	with open(outfile, 'a') as f:
		for j in range(i, i+N):
			np.savetxt(f, np.c_[j, halos[j].mass, halos[j].density[:,np.newaxis].T, halos[j].r200c], delimiter=',')
	
if verbose:
	begin = time.perf_counter()
	
[halo.get_density(box, x) for halo in halos[no_halos-(no_halos%N):]]

if verbose:
	end = time.perf_counter()
	print('------------------------')
	print(f'{no_halos} calculated')
	print(f'Time for {no_halos-(no_halos%N)}: {end-begin} s')
	print(f'Time for {no_halos}: {end-start} s')

	with open(outfile, 'a') as f:
		for j in range(no_halos-(no_halos%N), no_halos):
			np.savetxt(f, np.c_[j, halos[j].mass, halos[j].density[:,np.newaxis].T, halos[j].r200c], delimiter=',')
	
if verbose:
	print('All saved to file')
