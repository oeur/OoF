from oti_analysis import run_oti_analysis
from subselect_data import subselect_solar_cyls
from load_data import load_simulation_data
import matplotlib.patches as mpatches
from astropy.constants import G
from astropy import constants as const
import astropy.coordinates as coord
import astropy.table as at
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jaxopt
from statistics import mean
import scipy.interpolate as sci
from astropy.stats import median_absolute_deviation as MAD
from gala.units import UnitSystem, galactic
from jax.scipy.special import gammaln
from jaxopt import Bisection
from scipy.stats import binned_statistic, binned_statistic_2d
import torusimaging as oti
import sympy as sp
from scipy import stats
from itertools import product
import itertools
from matplotlib.colors import LogNorm
import time
from scipy.ndimage import gaussian_filter1d
import scipy
from scipy import stats
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
import gizmo_analysis as gizmo
import utilities as ut
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cmasher as cmr
from matplotlib.patheffects import withStroke
import requests
import pickle
import io

def compute_smd_stats(simdir, simnum, species, Rcyl, numvols, zcut):
    '''
    Generate plot of true FIRE & OTI-inferred stellar surface mass density at |z|=1.1 kpc.

    Args:
        simdir (str): filepath to directory where sim is located
        snapnum (int): snapshot number (e.g., 600)
        species (str): 'star', 'gas', 'dark' or 'all'
        Rcyl (float): Galactocentric radius (cylindrical) (e.g., 8)
        numvols (int): number of solar volumes (e.g., 16)
        zcut (float): value of the cut on |z|
    '''
    data_vols = subselect_solar_cyls(simdir, simnum, species, Rcyl, numvols, zcut)
    part_star = gizmo.gizmo_io.Read.read_snapshots([species], 'index', simnum, simulation_directory=simdir, assign_hosts_rotation=True, assign_hosts=True)
    x  = part_star[species].prop('host.distance.principal.cartesian')[:,0]
    y  = part_star[species].prop('host.distance.principal.cartesian')[:,1]
    z  = part_star[species].prop('host.distance.principal.cartesian')[:,2]
    age  = part_star[species].prop('age')
    feh  = part_star[species].prop('metallicity.fe')
    mgfe = part_star[species].prop('metallicity.mg - metallicity.fe')
    max_z_list = []

    for i in range(len(data_vols['x'])):
        z_array = data_vols['z'][i] * u.kpc
        max_z = np.round(3 * 1.5 * MAD(z_array), 1)
        
        max_z_list.append(max_z)
    res_list, bdata_list, model_list, bounds_list= run_oti_analysis(simdir, simnum, species, Rcyl, numvols, zcut)
    #MCMC uncertainty
    accs_mcmc = []
    for i in range(16):
        with open(f'./mcmc_2.5/volume-{i+1}-mcmc-results.pkl', 'rb') as file:
            mcmc_states, mcmc_params = pickle.load(file)

        accs = []
        zgrid = np.linspace(-1, 1, 1024) * max_z_list[i]
        for p in mcmc_params:
            acc = model_list[i].get_acceleration(zgrid, p)
            accs.append(acc.value)
        a_unit = u.km / u.s / u.Myr
        accs = accs * acc.unit
        accs_mcmc.append(accs.to(u.km / (u.Myr * u.s)).value)
        
        print(f'Finished processing (Vol. {i+1})')

    #bootstrapping uncertainty
    accs_boots = []
    for i in range(len(data_vols['x'])):
        with open(f'./boots_2.5/bootstrap_res_v{i+1}.pkl', 'rb') as file:
            bootstrap_params = pickle.load(file)

        accs = []
        zgrid = np.linspace(-1, 1, 1024) * max_z_list[i]

        for p in bootstrap_params:
            acc = model_list[i].get_acceleration(zgrid, p)
            accs.append(acc.value)
        
        accs = np.array(accs) * acc.unit
        accs_boots.append(accs.to(u.km / (u.Myr * u.s)).value)
    std_mcmc = [[np.std(accs_mcmc[j][:, i]) for i in range(1024)] for j in range(16)]
    std_boots = [[np.std(accs_boots[j][:, i]) for i in range(1024)] for j in range(16)]
    #Combine variances in quadrature to get a total uncertainty
    std_tot = [np.sqrt(np.array(u.Quantity(std_boots[i]))**2 + np.array(u.Quantity(std_mcmc[i]))**2) for i in range(16)]

    fire_az_binned = []
    bc = []

    for i in range(len(data_vols['vz'])):
    # Sort z_vols and accelerations arrays together based on z_vols
        sorted_indices = sorted(range(len(data_vols['z'][i])), key=lambda k: data_vols['z'][i][k])
        sorted_z_vols = [data_vols['z'][i][idx] for idx in sorted_indices]
        sorted_az_vols = [data_vols['az'][i][idx] for idx in sorted_indices]
        
        binned_az, bin_edges, binnumber = stats.binned_statistic(sorted_z_vols, sorted_az_vols, 'mean', bins=1024)
        fire_az_binned.append(binned_az)
        bin_width = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - bin_width/2
        bc.append(bin_centers)

    bestfit_az_list_bc = []

    for i in range(len(data_vols['x'])):
        bestfit_az = model_list[i].get_acceleration((bc[i]*u.kpc), res_list[i].params)
        bestfit_az_list_bc.append(bestfit_az)
        
    fire_az = (np.array(fire_az_binned) * u.km/ (u.s * u.Gyr)).to(u.km / (u.Myr * u.s))
    oti_arr = np.array(bestfit_az_list_bc)
    oti_az = (oti_arr* u.kpc/ (u.Myr)**2).to(u.km / (u.Myr * u.s))
    oti_az_in_pc = (oti_az.to(u.pc / (u.s)**2))
    fire_az_in_pc = (fire_az.to(u.pc / (u.s)**2))

    def compute_smd(a):
        G_in_pc = G.to(u.pc * (u.pc)**2 / (u.Msun * (u.s)**2))
        surface_mass_density = np.abs(a)/(2*np.pi*G_in_pc)
        return surface_mass_density

    fire_stellar_sigma = compute_smd(fire_az_in_pc)
    oti_stellar_sigma = compute_smd(oti_az_in_pc)
    std_in_pc = (std_tot*u.km / (u.Myr * u.s)).to(u.pc / (u.s)**2)

    def find_closest_index(array, value):
        array = np.asarray(array)
        index = (np.abs(array - value)).argmin()
        return index

    idx_above = [find_closest_index(bc[i], 1.1) for i in range(16)]
    idx_below = [find_closest_index(bc[i], -1.1) for i in range(16)]

    def compute_smd_oti_err(std):
        G_in_pc = G.to(u.pc * (u.pc)**2 / (u.Msun * (u.s)**2))
        smd_std = np.abs(std)/(2*np.pi*G_in_pc)
        return smd_std

    oti_sigma_std= compute_smd_oti_err(std_in_pc)

    fire_sigma_above = []
    oti_sigma_above = []
    sigma_std_above = []
    for i in range(len(data_vols['x'])):
        fire_sig = fire_stellar_sigma[i][idx_above[i]]
        oti_sig = oti_stellar_sigma[i][idx_above[i]]
        oti_sig_std = oti_sigma_std[i][idx_above[i]]
        fire_sigma_above.append(fire_sig)
        oti_sigma_above.append(oti_sig)
        sigma_std_above.append(oti_sig_std)
        
    fire_sigma_below = []
    oti_sigma_below = []
    sigma_std_below = []
    for i in range(len(data_vols['x'])):
        fire_sig = fire_stellar_sigma[i][idx_below[i]]
        oti_sig = oti_stellar_sigma[i][idx_below[i]]
        oti_sig_std = oti_sigma_std[i][idx_below[i]]
        fire_sigma_below.append(fire_sig)
        oti_sigma_below.append(oti_sig)
        sigma_std_below.append(oti_sig_std)
    fire_above = [q.value for q in fire_sigma_above]
    fire_below = [q.value for q in fire_sigma_below]
    total_sum = 0

    for i in [0, 1, 2, 12, 13, 14, 15]:
        total_sum += np.sum(fire_above[i])

    avg_sigma_bad_vols_above = total_sum / 7
    print(avg_sigma_bad_vols_above)
    total_sum = 0

    for i in [3, 4, 5, 6, 7, 8, 9, 10, 11]:
        total_sum += np.sum(fire_above[i])

    avg_sigma_good_vols_above = total_sum / 9
    print(avg_sigma_good_vols_above)
    total_sum = 0

    for i in [0, 1, 2, 12, 13, 14, 15]:
        total_sum += np.sum(fire_below[i])

    avg_sigma_bad_vols_below = total_sum / 7
    print(avg_sigma_bad_vols_below)
    total_sum = 0

    for i in [3, 4, 5, 6, 7, 8, 9, 10, 11]:
        total_sum += np.sum(fire_below[i])

    avg_sigma_good_vols_below = total_sum / 9
    print(avg_sigma_good_vols_below)