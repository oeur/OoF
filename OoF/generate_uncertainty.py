from astropy.constants import G
from astropy import constants as const
import astropy.coordinates as coord
import astropy.table as at
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jaxopt
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
# gala
import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
#FIRE
import gizmo_analysis as gizmo
import utilities as ut
import pickle
import time
from pathlib import Path
from .oti_analysis import run_oti_analysis
from .subselect_data import subselect_solar_cyls
def generate_mcmc_results(simdir, simnum, species, Rcyl, numvols, zcut, file_path="."):
    """
    Run MCMC for a given set of volumes and save the results to files.

    Args:
        simdir (str): filepath to directory where sim is located
        snapnum (int): snapshot number (e.g., 600)
        species (str): 'star', 'gas', 'dark' or 'all'
        Rcyl (float): Galactocentric radius (cylindrical) (e.g., 8)
        numvols (int): number of solar volumes (e.g., 16)
        zcut (float): value of the cut on |z|
        file_path (str): Path to save the result files.
    """
    res_list, bdata_list, model_list, bounds_list = run_oti_analysis(simdir, simnum, species, Rcyl, numvols, zcut) #zcut=30
    file_path = Path(file_path)

    for volume in range(numvols):
        # Save parameters to file
        with open(file_path / f"volume-{volume + 1}-params-opt.pkl", "wb") as f:
            pickle.dump(res_list[volume].params, f)

        # Run MCMC
        start_time = time.time()
        print(f"Running MCMC for volume {volume + 1}...")
        states, mcmc_samples = model_list[volume].mcmc_run_label(
            bdata_list[volume], p0=res_list[volume].params, bounds=bounds_list[volume], num_warmup=1000, num_steps=1000
        )
        
        # Save MCMC results to a file
        with open(file_path / f"volume-{volume + 1}-mcmc-results.pkl", "wb") as f:
            pickle.dump((states, mcmc_samples), f)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        #print(f"Cell execution time for volume {volume + 1}: {elapsed_time} seconds")

def generate_bootstrap_resamplings(simdir, simnum, species, Rcyl, numvols, zcut, trials, file_path="."): #trials=128
    """
    Perform bootstrapping for a set of volumes and save the results to files.

    Args:
        simdir (str): filepath to directory where sim is located
        snapnum (int): snapshot number (e.g., 600)
        species (str): 'star', 'gas', 'dark' or 'all'
        Rcyl (float): Galactocentric radius (cylindrical) (e.g., 8)
        numvols (int): number of solar volumes (e.g., 16)
        zcut (float): value of the cut on |z|
        file_path (str): Path to save the result files.
    """
    data_vols = subselect_solar_cyls(simdir, simnum, species, Rcyl, numvols, zcut) #zcut=10
    file_path = Path(file_path)
    z_array_list = []
    vz_array_list = []
    max_z_list = []
    bdata_list = []
    idx_list = [0]
    for i in range(len(data_vols['z'])):
        z_array = data_vols['z'][i] * u.kpc
        vz_array = data_vols['vz'][i] * (u.km / u.s)

        z_array_list.append(z_array)
        vz_array_list.append(vz_array)

        max_z = np.round(3 * 1.5 * MAD(z_array), 1)
        max_vz = np.round(3 * 1.5 * MAD(vz_array), 0)

        max_z_list.append(max_z)

        zvz_bins = {
            "pos": np.linspace(-max_z, max_z, 101),
            "vel": np.linspace(-max_vz, max_vz, 101),
        }

        bdata = oti.data.get_binned_label(
            z_array,
            vz_array,
            label=data_vols['feh'][i],
            bins=zvz_bins,
            units=galactic,
            s_N_thresh=32,
        )
        bdata_list.append(bdata)

    model_list = []
    bounds_list = []
    init_params_list = []
    for i in range(len(data_vols['z'])):
        model, bounds, init_params = oti.TorusImaging1DSpline.auto_init(
            bdata_list[i],
            label_knots=8,
            e_knots={2: 10, 4: 5},
            label_l2_sigma=1.0,
            label_smooth_sigma=0.5,
            e_l2_sigmas={2: 1.0, 4: 1.0},
            e_smooth_sigmas={2: 0.1, 4: 0.1},
            dacc_strength=0.0,
            label_knots_spacing_power=0.75,
            e_knots_spacing_power=0.75,
        )

        init_params["e_params"][2]["vals"] = np.full_like(
            init_params["e_params"][2]["vals"], -0.5
        )
        init_params["e_params"][4]["vals"] = np.full_like(
            init_params["e_params"][4]["vals"], np.log(0.05 / model._label_knots.max())
        )

        model_list.append(model)
        bounds_list.append(bounds)
        init_params_list.append(init_params)

    data_kw_list = []
    mask_list = []

    for i in range(len(data_vols['z'])):
        data_kw = dict(
            pos=bdata_list[i]["pos"],
            vel=bdata_list[i]["vel"],
            label=bdata_list[i]["label"],
            label_err=bdata_list[i]["label_err"],
        )

        mask = (
            np.isfinite(bdata_list[i]["label"])
            & np.isfinite(bdata_list[i]["label_err"])
            & (bdata_list[i]["label_err"] > 0)
        )

        data_kw = {k: v[mask] for k, v in data_kw.items()}

        data_kw_list.append(data_kw)
        mask_list.append(mask)

    rng = np.random.default_rng(seed=42)

    for volume in range(numvols):
        start_time = time.time()
        
        bootstrap_res = []

        for trial in range(trials):
            idx = rng.choice(len(data_kw_list[volume]["label"]), size=len(data_kw_list[volume]["label"]), replace=True)
            bdata_trial = {k: v[idx] for k, v in data_kw_list[volume].items()}

            res = model_list[volume].optimize(init_params_list[volume], objective="gaussian", bounds=bounds_list[volume], **bdata_trial)
            print(f"Volume {volume+1}, Trial {trial}: Success={res.state.success}, Iterations={res.state.iter_num}")
            bootstrap_res.append(res)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        #print(f"Volume {volume+1} execution time: {elapsed_time} seconds")

        with open(file_path / f"bootstrap_res_v{volume+1}.pkl", "wb") as f:
            data_to_save = [res.params for res in bootstrap_res]
            pickle.dump(data_to_save, f)
# simpath = '/Users/micahoeur/Dropbox/Research/Sarah/FIRE2_m12i_metal_diffusion/output_with_accel' #/scratch/07439/moeur/GalaxiesOnFIRE/metal_diffusion/m12i_r7100/output_with_accel
# snum = 600
# spcs = 'star'
# z_cut = 2.5
# generate_mcmc_results(simpath, snum, spcs, 8, 16, z_cut, file_path=".")
# generate_bootstrap_resamplings(simpath, snum, spcs, 8, 16, z_cut, 128, file_path=".")
