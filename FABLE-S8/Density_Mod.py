import numpy as np
import h5py
import pandas as pd
import time
from pathlib import Path
import functions as fn
import Box
import Halo

#variables
z = 0
x = np.logspace(-2, 0, 21)
verbose = True
#consts
halos = np.array([])
boxes = np.array([])

#create boxes and halos
for model in fn.models:
	outfile = fn.here_path/'Density_Profiles'/f'{model}_z{z}.csv'
	model_path = fn.models_path/model
	fof_path = model_path/f'fof_subhalo_tab_{fn.zhot[z]}.hdf5'

	box = Box.Box(model_path, fn.zhot[z])

	if verbose:
		print('----------------------------')
		print(f'{model}, snapshot {fn.zhot[z]}, z={box.z}')

	with h5py.File(fof_path) as fof:
		mass = np.array(fof.get('Group/GroupMass'))
		ndx = np.where((mass==np.amax(mass)))
		halo_pos = np.array(fof.get('Group/GroupPos'))[ndx]
		halo_r200c = np.array(fof.get('Group/Group_R_Crit200'))[ndx]

		halo = Halo.Halo(halo_pos, halo_r200c)

	#get density profiles

	if verbose:
		print('Calculating density profiles')
		start = time.perf_counter()

	halo.get_density(box, x)

	if verbose:
		end = time.perf_counter()
		print(f'Density profile calculated in {end-start} s')
		print('Saving to file')

	#save to file
	with open(outfile, 'w') as f:
		header = 'i, M,'
		for i, _ in enumerate(x[:-1]):
			header += f'rho_{i},'
		header += 'r200\n'
		f.write(header)
		np.savetxt(f, np.c_[0, halo.mass, halo.density[:,np.newaxis].T, halo.r200c], delimiter=',')
		
	if verbose:
		print('Saved to file')
