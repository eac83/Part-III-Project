"""Class Box: Stores information about the box (from snap file)."""
import numpy as np
import h5py


class Box():

    """Store information about the the box."""

    # pylint: disable=too-many-instance-attributes
    # box and particles have more than seven attributes.
    def __init__(self, box_path, shot):
        """Use given box and snapshot to assign variables"""
        snap_path = box_path/f'snap_{shot}.hdf5'

        with h5py.File(snap_path) as snap:
            self.z = snap['Header'].attrs['Redshift']
            self.size = snap['Header'].attrs['BoxSize']
            self.h = snap['Header'].attrs['HubbleParam']
            self.omega_m = snap['Header'].attrs['Omega0']

            self.coords_gas = np.array(snap.get('PartType0/Coordinates'))
            self.coords_dm = np.array(snap.get('PartType1/Coordinates'))
            self.coords_stars = np.array(snap.get('PartType4/Coordinates'))

            self.mass_gas = np.array(snap.get('PartType0/Masses'))
            self.mass_dm = np.array([snap['Header'].attrs['MassTable'][1]] *
                                    self.coords_dm.shape[0])
            self.mass_stars = np.array(snap.get('PartType4/Masses'))
