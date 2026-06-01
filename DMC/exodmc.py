#!/usr/bin/env python
# coding: utf-8

__author__ = 'Mariangela Bonavita'
__version__ = 'v2.0.1'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy import interpolate
from scipy.special import erf
from numpy import random as rn
import copy
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


##########################################
###              EXO-DMC               ###
### EXOplanet Detection Map Calculator ###
##########################################


class exodmc(object):

    def __init__(self, star_ID, star_dist, **kwargs):

        if not isinstance(star_ID, list):
            star_ID = [star_ID]

        if not isinstance(star_dist, list):
            star_dist = [star_dist]

        self.ID = star_ID
        self.dpc = star_dist  # pc

        self.set_grid(**kwargs)

    ##########################################################################
    # GRID SETUP
    ##########################################################################

    def set_grid(
        self,
        x_min=0.1,
        x_max=1000.,
        nx=100,
        logx=False,
        y_min=0.1,
        y_max=100.,
        ny=100,
        logy=False,
        ngen=1000,
        e_params={'shape': 'gauss', 'mean': 0, 'sigma': 0.3},
        i_params={'shape': 'cos_i'},
        rho_visibility=None
    ):

        ######################################################################
        # STORE PARAMETERS
        ######################################################################

        self.x_min = x_min
        self.x_max = x_max
        self.x_nsteps = nx
        self.logx = logx

        self.y_min = y_min
        self.y_max = y_max
        self.y_nsteps = ny
        self.logy = logy

        self.norb = ngen

        ######################################################################
        # ECCENTRICITY DISTRIBUTION
        ######################################################################

        self.e_dist = e_params['shape']

        self.e_mu = e_params.get('mean', 0)
        self.e_sigma = e_params.get('sigma', 0.3)

        self.e_min = e_params.get('min', 0)
        self.e_max = e_params.get('max', 1)

        ######################################################################
        # INCLINATION DISTRIBUTION
        ######################################################################

        self.i_dist = i_params['shape']

        self.i_mu = i_params.get('mean', 0)
        self.i_sigma = i_params.get('sigma', 1)

        ######################################################################
        # SMA GRID
        ######################################################################

        if logx:
            self.sma = np.logspace(
                np.log10(x_min),
                np.log10(x_max),
                nx
            )
        else:
            self.sma = np.linspace(x_min, x_max, nx)

        ######################################################################
        # MASS GRID
        ######################################################################

        if logy:
            self.M2 = np.logspace(
                np.log10(y_min),
                np.log10(y_max),
                ny
            )
        else:
            self.M2 = np.linspace(y_min, y_max, ny)

        ######################################################################
        # GENERATE ECCENTRICITIES
        ######################################################################

        if self.e_dist == 'gauss':

            self.ecc = cropped_gaussian(
                self.e_mu,
                self.e_sigma,
                self.norb,
                limits=[self.e_min, self.e_max]
            )

        elif self.e_dist == 'uniform':

            self.ecc = (
                self.e_min +
                (self.e_max - self.e_min)
                * rn.random_sample(self.norb)
            )

        else:
            raise ValueError(
                "e_params['shape'] must be 'gauss' or 'uniform'."
            )

        ######################################################################
        # RANDOM ORBITAL ANGLES
        ######################################################################

        self.Omega_Node = rn.random_sample(self.norb) * 2. * np.pi
        self.Omega_Peri = rn.random_sample(self.norb) * 2. * np.pi

        self.omega = self.Omega_Peri - self.Omega_Node

        ######################################################################
        # INCLINATIONS
        ######################################################################

        if self.i_dist == 'cos_i':

            cosi = 2 * rn.random_sample(self.norb) - 1.
            self.irad = np.arccos(cosi)

        elif self.i_dist == 'gauss':

            self.irad = np.abs(
                rn.normal(self.i_mu, self.i_sigma, self.norb)
            )

        else:
            raise ValueError(
                "i_params['shape'] must be 'gauss' or 'cos_i'."
            )

        ######################################################################
        # MEAN ANOMALY
        ######################################################################

        self.M = rn.random_sample(self.norb) * 2 * np.pi

        ######################################################################
        # ECCENTRIC ANOMALY
        ######################################################################

        self.E1 = iter_eccentric_anomaly(self.M, self.ecc)

        ######################################################################
        # TRUE ANOMALY
        ######################################################################

        ecc_expr = (1 + self.ecc) / (1 - self.ecc)

        self.nurad = 2 * np.arctan(
            np.sqrt(ecc_expr) * np.tan(self.E1 / 2.)
        )

        ######################################################################
        # THIELE-INNES ELEMENTS
        ######################################################################

        A1 = (
            np.cos(self.Omega_Peri) * np.cos(self.Omega_Node)
            - np.sin(self.Omega_Peri)
            * np.sin(self.Omega_Node)
            * np.cos(self.irad)
        )

        B1 = (
            np.cos(self.Omega_Peri) * np.sin(self.Omega_Node)
            + np.sin(self.Omega_Peri)
            * np.cos(self.Omega_Node)
            * np.cos(self.irad)
        )

        F1 = (
            -np.sin(self.Omega_Peri) * np.cos(self.Omega_Node)
            - np.cos(self.Omega_Peri)
            * np.sin(self.Omega_Node)
            * np.cos(self.irad)
        )

        G1 = (
            -np.sin(self.Omega_Peri) * np.sin(self.Omega_Node)
            + np.cos(self.Omega_Peri)
            * np.cos(self.Omega_Node)
            * np.cos(self.irad)
        )

        ######################################################################
        # ORBITAL POSITIONS
        ######################################################################

        x1 = np.cos(self.E1) - self.ecc[np.newaxis, :]

        y1 = (
            np.sqrt(1 - self.ecc[np.newaxis, :]**2)
            * np.sin(self.E1)
        )

        y2 = B1[np.newaxis, :] * x1 + G1[np.newaxis, :] * y1
        x2 = A1[np.newaxis, :] * x1 + F1[np.newaxis, :] * y1

        ######################################################################
        # PROJECTED SEPARATION
        ######################################################################

        self.rad = np.sqrt(x2**2 + y2**2).T

        self.rho = (
            self.rad[:, np.newaxis]
            * (self.sma[:, np.newaxis] / self.dpc)
        ).T

        ######################################################################
        # OPTIONAL VISIBILITY CORRECTION
        ######################################################################

        if rho_visibility is not None:

            if not isinstance(rho_visibility, dict):
                raise TypeError(
                    'rho_visibility must be a dictionary.'
                )

            if (
                'separation' not in rho_visibility
                or
                'visibility' not in rho_visibility
            ):
                raise TypeError(
                    'rho_visibility must contain '
                    '"separation" and "visibility".'
                )

            vis_f = interpolate.interp1d(
                rho_visibility['separation'],
                rho_visibility['visibility'],
                bounds_error=False,
                fill_value=(1, 0)
            )

            self.rho_visibility = vis_f(self.rho)

    ##########################################################################
    # DIRECT IMAGING MODE
    ##########################################################################

    def DImode(
        self,
        xlim,
        ylim,
        lxunit='as',
        lyunit='Mjup',
        verbose=True,
        plot=True,
        savefig=True
    ):

        ns = np.size(self.dpc)

        detmap = []
        self.detflag = []

        if not isinstance(xlim, np.ndarray):
            xlim = np.array(xlim)

        if not isinstance(ylim, np.ndarray):
            ylim = np.array(ylim)

        if ns == 1:
            xlim = [xlim]
            ylim = [ylim]

        ######################################################################
        # LOOP OVER TARGETS
        ######################################################################

        for ll in range(ns):

            start = time.time()

            det = np.zeros(
                (self.x_nsteps, self.y_nsteps, self.norb)
            )

            ##################################################################
            # UNIT CONVERSION
            ##################################################################

            if lxunit == 'au':
                xlim[ll] = xlim[ll] / self.dpc[ll]

            if lxunit == 'mas':
                xlim[ll] = xlim[ll] / 1000.

            ##################################################################
            # VISIBILITY
            ##################################################################

            if hasattr(self, 'rho_visibility'):
                values = self.rho_visibility[ll]
            else:
                values = np.ones_like(self.rho[0])

            ##################################################################
            # VALID MASS RANGE
            ##################################################################

            #valid = np.where(ylim[ll] < self.y_max)[0]

            #if len(valid) == 0:
            #    raise ValueError(
            #        "No valid masses found within y_max."
            #    )

            #max_mass = np.nanmax(ylim[ll][valid])
            #min_mass = np.nanmin(ylim[ll][valid])

            ##################################################################
            # INTERPOLATE DETECTION LIMIT
            ##################################################################

            rlim = np.interp(
                self.rho[ll],
                xlim[ll],
                ylim[ll],
                right=np.nan,
                left=np.nan
            )

            ##################################################################
            # MAIN LOOP
            ##################################################################

            for i in range(self.x_nsteps):

                outside = np.where(
                    (self.rho[ll, i] < np.min(xlim[ll]))
                    |
                    (self.rho[ll, i] > np.max(xlim[ll]))
                )[0]

                for j in range(self.y_nsteps):

                    ##################################################################
                    # MASS ABOVE DETECTION LIMIT
                    ##################################################################

                    #if min_mass < self.M2[j] < max_mass:

                    #	detectable = np.where(rlim[i] < self.M2[j])[0]
                    

                    #    if len(detectable) > 0:

                    #        det[i, j, detectable] = (
                    #            values[i, detectable]
                    #        )
                    
                    detectable = np.where((self.M2[j] > rlim[i]) & np.isfinite(rlim[i]))[0]

                    if len(detectable) > 0:

                    	det[i, j, detectable] = values[i, detectable]

                    ##################################################################
                    # FORCE OUTSIDE-FOV TO ZERO
                    ##################################################################

                    if len(outside) > 0:

                        det[i, j, outside] = 0

            ##################################################################
            # EXTRAPOLATE TO HIGH MASSES
            ##################################################################

            #high_mass = np.where(self.M2 > max_mass)[0]

            #if len(high_mass) > 0:

            #   det[:, high_mass, :] = np.tile(
            #        det[:, high_mass[0]-1, :][:, np.newaxis, :],
            #        [1, len(high_mass), 1]
            #    )

            ##################################################################
            # DETECTION PROBABILITY
            ##################################################################

            det_probability = np.sum(det, axis=2) / self.norb

            detmap.append(det_probability)
            self.detflag.append(det)

            ##################################################################
            # TIMING
            ##################################################################

            end = time.time()

            hours, rem = divmod(end-start, 3600)
            minutes, seconds = divmod(rem, 60)

            if verbose:

                print(
                    self.ID[ll],
                    "time elapsed - "
                    "{:0>2}:{:0>2}:{:05.2f}".format(
                        int(hours),
                        int(minutes),
                        seconds
                    )
                )

            ##################################################################
            # PLOT
            ##################################################################

            if plot:

                fig = plt.figure(figsize=(8, 6))

                ax = fig.add_axes([0.15, 0.15, 0.8, 0.7])

                ax.set_xscale('log')
                ax.set_yscale('log')

                ax.set_ylabel("Mass (M$_{Jup}$)")
                ax.set_xlabel("Semi-major axis (AU)")

                levels = [10, 20, 50, 70, 90, 95, 99]#, 100]

                norm = mcolors.Normalize(0, 100)

                cf0 = ax.contourf(
                    self.sma,
                    self.M2,
                    det_probability.T * 100,
                    norm=norm,
                    levels=np.arange(0, 100.01, 0.1),
                    cmap='Blues'
                )

                contours = plt.contour(
                    self.sma,
                    self.M2,
                    det_probability.T * 100,
                    levels,
                    cmap='bone',
                    linewidths=1
                )

                CB = plt.colorbar(
                    cf0,
                    ticks=levels
                )

                CB.add_lines(contours)

                CB.ax.set_yticklabels(
                    [r"{:.0f}$\%$".format(i)
                     for i in CB.get_ticks()]
                )

                plt.title(self.ID[ll])

                if savefig:
                    plt.savefig(
                        self.ID[ll] + '_detprob.png',
                        dpi=300
                    )

        return detmap


##############################################################################
# ECCENTRIC ANOMALY SOLVER
##############################################################################

def iter_eccentric_anomaly(M, e, tol=1e-10, nIter=100):

    E = copy.deepcopy(M)

    for _ in range(nIter):

        delta = (
            (M - (E - e*np.sin(E)))
            /
            (1 - e*np.cos(E))
        )

        E += delta

        if np.max(np.abs(delta)) < tol:
            break

    return E


##############################################################################
# CROPPED GAUSSIAN
##############################################################################

def cropped_gaussian(mu, sigma, norb, limits=[0, 1]):

    """
    Defines a set of nord points, distributed according to a cropped Gaussian defined
    by (mu, sigma) and bound between two boundaries specified by the list "limits".
    Unlike mirroring (e.g., np.abs()) or flooring (np.crop()) function, this function
    retains the Gaussianity of the returned array.
    Credits: Vito Squicciarini
    """

    if (mu < limits[0]) or (mu > limits[1]):

        raise ValueError(
            f'mu must be between {limits[0]} and {limits[1]}'
        )

    if sigma <= 0:

        return np.ones(norb) * mu

    if sigma > 5 * (limits[1] - limits[0]):

        return np.ones(norb) / (limits[1] - limits[0])

    Phi = lambda x, mu, sigma: (
        0.5 * (
            1 + erf(
                (x - mu)
                /
                (np.sqrt(2) * sigma)
            )
        )
    )

    expected_invalid = (
        Phi(limits[0], mu, sigma)
        +
        (1 - Phi(limits[1], mu, sigma))
    )

    scaling_factor = int(
        np.ceil(2 / (1 - expected_invalid))
    )

    a1 = rn.normal(
        mu,
        sigma,
        scaling_factor * norb
    )

    a1 = a1[
        (a1 > limits[0])
        &
        (a1 < limits[1])
    ]

    return a1[:norb]
