'''Calculate density profiles of FABLE zoom-in haloes'''
import time
import numpy as np
import h5py
import functions as fn
import box
import halo

#variables and consts
z = 0
x_BINS = np.logspace(-2, 0, 21)
IS_VERBOSE = True
OUTFILE = fn.HERE_PATH/'Density_Profiles'/f'zooms_z{z}.csv'
halos = np.array([])
boxes = np.array([])

#create boxes and halos
for i in range(96, 513, 16):
    if i == 416:
        f_no = '410'
    else:
        f_no = str(i)


    #grab correct snapshot
    if (fn.ZOOMS_PATH/f'c{f_no}_MDRInt'
        /f'fof_subhalo_tab_{fn.XHOT[0]}.hdf5').exists():
        shot = fn.XHOT[z]
    elif (fn.ZOOMS_PATH/f'c{f_no}_MDRInt'
       /f'fof_subhalo_tab_{fn.ZHOT[0]}.hdf5').exists():
        shot = fn.ZHOT[z]
    else:
        shot = fn.SHOT[z]

    #create box and halo
    zoom_path = fn.ZOOMS_PATH/f'c{f_no}_MDRInt'
    fof_path = zoom_path/f'fof_subhalo_tab_{shot}.hdf5'

    boxes = np.append(boxes, box.Box(zoom_path, shot))

    if IS_VERBOSE:
        print(f'Zoom c{f_no}, snapshot {shot}, z={boxes[-1].z}')
    with h5py.File(fof_path) as fof:
        mass = np.array(fof.get('Group/GroupMass'))
        ndx = np.where((mass == np.amax(mass)))
        halo_pos = np.array(fof.get('Group/GroupPos'))[ndx]
        halo_r200c = np.array(fof.get('Group/Group_R_Crit200'))[ndx]

        halos = np.append(halos, halo.Halo(halo_pos, halo_r200c))

#get density profiles
if IS_VERBOSE:
    print('Calculating density profiles')
    START = time.perf_counter()

[halo_i.get_density(boxes[i], x_BINS) for i, halo_i in enumerate(halos)]

if IS_VERBOSE:
    END = time.perf_counter()
    print(f'Density profiles calculated in {END-START} s')
    print('Saving to file')

#save to file
with open(OUTFILE, 'w') as f:
    header = 'i, M,'
    for i, _ in enumerate(x_BINS[:-1]):
        header += f'rho_{i},'
    header += 'r200\n'
    f.write(header)
    for halo in halos:
        np.savetxt(f, np.c_[i, halo.mass, halo.density[:,np.newaxis].T,
                      halo.r200c], delimiter=',')

if IS_VERBOSE:
    print('Saved to file')
