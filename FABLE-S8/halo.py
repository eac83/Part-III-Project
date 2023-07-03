"""Class Halo, store details about a given halo."""
import numpy as np
import functions as fn


class Halo():

    """Store information about a halo."""
    #pylint: disable=too-many-instance-attributes
    #Halos in Illustris-type sims have more than seven attributes.
    def __init__(self, pos, r200c):
        """Assign pos and r_200, initialize everything else."""
        self.pos = pos
        self.r200c = r200c
        self.mass_gas = np.array([])
        self.mass_dm = np.array([])
        self.mass_stars = np.array([])
        self.mass = 0
        self.density_gas = np.array([])
        self.density_dm = np.array([])
        self.density_stars = np.array([])
        self.density = np.array([])
        self.u = np.array([])

    def __str__(self):
        """Print halo's pos and r_200.'"""
        return f'Pos: {self.pos}, R200c: {self.r200c}'

    def get_density(self, box, x_bins):
        """Assign halo's density profiles in the given x bins."""
        disp_gas = fn.nearest(box.coords_gas-self.pos, box.size)
        disp_dm = fn.nearest(box.coords_dm-self.pos, box.size)
        disp_stars = fn.nearest(box.coords_stars-self.pos, box.size)

        x_gas = np.sqrt(np.sum(disp_gas**2, axis=1)) / self.r200c
        x_dm = np.sqrt(np.sum(disp_dm**2, axis=1)) / self.r200c
        x_stars = np.sqrt(np.sum(disp_stars**2, axis=1)) / self.r200c

        self.mass_gas = np.array([])
        self.mass_dm = np.array([])
        self.mass_stars = np.array([])

        for i, x_i in enumerate(x_bins[:-1]):
            gdx = np.where((x_gas >= x_i) & (x_gas < x_bins[i+1]))
            self.mass_gas = np.append(self.mass_gas, np.sum(box.mass_gas[gdx]))
            ddx = np.where((x_dm >= x_i) & (x_dm < x_bins[i+1]))
            self.mass_dm = np.append(self.mass_dm, np.sum(box.mass_dm[ddx]))
            sdx = np.where((x_stars >= x_i) & (x_stars < x_bins[i+1]))
            self.mass_stars = np.append(self.mass_stars,
                                        np.sum(box.mass_stars[sdx]))

        self.density_gas = self.mass_gas / (4 * np.pi / 3 *
                                            (x_bins[1:]**3 - x_bins[:-1]**3)
                                            * self.r200c**3)
        self.density_dm = self.mass_dm / (4 * np.pi / 3 *
                                          (x_bins[1:]**3 - x_bins[:-1]**3)
                                          * self.r200c**3)
        self.density_stars = self.mass_stars / (4 * np.pi / 3 *
                                                (x_bins[1:]**3 -
                                                 x_bins[:-1]**3)
                                                * self.r200c**3)

        self.mass = np.sum(self.mass_gas + self.mass_dm + self.mass_stars)
        self.density = self.density_gas + self.density_dm + self.density_stars

    def get_u(self, x, k):
        """Assign Fourier transformed density profile."""
        r = x * self.r200c
        norm = np.trapz(r**2 * self.density, r)
        self.u = np.array([])

        for k_i in k:
            self.u = np.append(self.u, np.trapz(self.density * r**2 *
                                                (np.sin(k_i*r) / (k_i*r)), r)
                               / norm)

    def slope(self, x, a, b):
        """Take in two slopes a and b, transform density profile according to
        the fixed-slope method."""
        mass = np.array([])
        self.mass = 4*np.pi*np.trapz(x**2*self.density, x)

        for i, x_pivot in enumerate(x[1:-1]):
            density = self.density.copy()
            density[:i+1] *= x_pivot**-a * x[:i+1]**a
            density[i+1:] *= x_pivot**-b * x[i+1:]**b
            mass = np.append(mass, 4 * np.pi * np.trapz(x**2 * density, x))

        ndx = np.where((abs((1 - mass/self.mass))
                        == np.amin(abs(1 - mass/self.mass))))[0][0]
        self.density[:ndx+1] *= x[ndx+1]**-a * x[:ndx+1]**a
        self.density[ndx+1:] *= x[ndx+1]**-b * x[ndx+1:]**b
        self.mass = (4 * np.pi * np.trapz(x**2 * self.density, x) *
                     self.r200c**3 * 1e10)

    def pivot(self, x, x_pivot):
        """Take in a pivot point, transform density profile according to the
        fixed-pivot point method."""
        mass = np.array([])
        slopes = np.array([])
        self.mass = 4 * np.pi * np.trapz(x**2 * self.density, x)
        ndx = np.where(x == x_pivot)[0][0]

        for a in [k/100 for k in range(1, 400)]:
            for b in [k/100 for k in range(0, 400)]:
                density = self.density.copy()
                density[:ndx] *= (x[:ndx] / x_pivot)**a
                density[ndx:] *= (x[ndx:] / x_pivot)**b
                mass = np.append(mass, 4 * np.pi * np.trapz(x**2 * density, x))
                slopes = np.append(slopes, (a, b))

        slopes = np.reshape(slopes, (2, int(slopes.shape[0] / 2)))
        mdx = np.where((abs(1 - mass/self.mass)
                        == np.amin(abs(1 - mass/self.mass))))[0][0]
        slope = slopes[:, mdx]
        self.density[:ndx] *= (x[:ndx] / x_pivot)**slope[0]
        self.density[ndx:] *= (x[ndx:] / x_pivot)**slope[1]
