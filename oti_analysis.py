from subselect_data import subselect_solar_cyls
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
from matplotlib.colors import LogNorm
import time
from scipy.ndimage import gaussian_filter1d
import scipy
from scipy import stats
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
import gizmo_analysis as gizmo
import utilities as ut
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmasher as cmr
#%matplotlib inline

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'font.size': 20})

plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

def run_oti_analysis(simdir, simnum, species, zcut):
    data_vols = subselect_solar_cyls(simdir, simnum, species, zcut)
    z_array_list = []
    vz_array_list = []
    max_z_list = []
    bdata_list = []

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
        model, bounds, init_params = oti.TorusImaging1DSpline.auto_init( # initializes the model
            bdata_list[i], #The current data segment to be processed
            label_knots=8, #Number of knots for the labels, a parameter related to spline interpolation
            e_knots={2: 10, 4: 5}, #Specifies the number of knots for e_fns
            label_l2_sigma=1.0, #Regularization parameter for labels
            label_smooth_sigma=0.5, #Smoothing parameter for labels
            e_l2_sigmas={2: 1.0, 4: 1.0}, #Regularization parameters for e_fns
            e_smooth_sigmas={2: 0.1, 4: 0.1}, #Smoothing parameters for e_fns
            dacc_strength=0.0, 
            label_knots_spacing_power=0.75, #Power for spacing of label knots
            e_knots_spacing_power=0.75, #Power for spacing of e_fn knots
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

    res_list = []

    for i in range(len(data_vols['z'])):
        res = model_list[i].optimize(init_params_list[i], objective="gaussian", bounds=bounds_list[i], **data_kw_list[i])
        res_list.append(res)

    return res_list

def plot_oti_results(simdir, simnum, species, zcut):
    res_list = run_oti_analysis(simdir, simnum, species, zcut)
    for i in range(len(data_vols['z'])):
        fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharex=True, sharey=True, constrained_layout=True)
        
        cs = axes[0].pcolormesh(
        bdata_list[i]["vel"].to_value(u.km/u.s),
        bdata_list[i]["pos"].to_value(u.kpc),
        bdata_list[i]["label"],
        vmin=-1.75,
        vmax=0,
        cmap=cmr.torch
        )
        cb = fig.colorbar(cs, ax=axes[0:2],fraction=0.023, pad=0.02)
        cb.set_label(r"$\langle$[Fe/H]$\rangle$ [dex]", fontsize=25, rotation=270, labelpad=30)
        cb.ax.set_ylim(-1.75, 0)
        cb.ax.yaxis.set_tick_params(labelsize=25)
        axes[0].set_aspect(40)
        axes[0].set_ylim(-3.5, 3.5)
        axes[0].set_xlim(-150, 150)
        
        model_feh = model_list[i].get_label(bdata_list[i]["pos"], bdata_list[i]["vel"], res_list[i].params)
        cs = axes[1].pcolormesh(
            bdata["vel"].to_value(u.km / u.s),
            bdata["pos"].to_value(u.kpc),
            model_feh,
            cmap=cmr.torch,
            rasterized=True,
            vmin=-1.75,
            vmax=0,
        )
        axes[1].set_aspect(40)
        axes[1].set_ylim(-3.5, 3.5)
        axes[1].set_xlim(-150, 150)
        
        cs = axes[2].pcolormesh(
            bdata["vel"].to_value(u.km / u.s),
            bdata["pos"].to_value(u.kpc),
            (bdata["label"] - model_feh) / bdata["label_err"],
            cmap="RdBu_r",
            vmin=-3,
            vmax=3,
            rasterized=True,
        )
        cb = fig.colorbar(cs, ax=axes[2],fraction=0.046, pad=0.04)
        cb.set_label("(FIRE $-$ OTI) / error", fontsize=25, rotation=270, labelpad=30)
        cb.ax.yaxis.set_tick_params(labelsize=25)
        axes[2].set_aspect(40)
        axes[0].set_ylabel(f"$z$ [{u.kpc:latex_inline}]", fontsize=25)
        for ax in axes:
            ax.set_xlabel(f"$v_z$ [{u.km/u.s:latex_inline}]", fontsize=25)
            ax.tick_params(axis='both', which='major', labelsize=25)
            
        axes[0].set_title(f"FIRE Data V{i+1}", fontsize=25)
        axes[1].set_title("OTI Fitted Model", fontsize=25)
        axes[2].set_title("Normalized Residuals", fontsize=25)
    plt.show() 