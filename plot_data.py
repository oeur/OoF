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
import cmasher as cmr
from matplotlib.patheffects import withStroke
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'font.size': 20})

plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

def generate_surface_mass_density_plot(simdir, simnum, species1, species2, Rcyl, numvols, zcut): #use zcut=1.5
    '''
    Generate 2-panel surface mass density plot of FIRE gas and stellar data.
    simdir (str): filepath to directory where sim is located
    snapnum (int): snapshot number (e.g., 600)
    species1 (str): first particle species, i.e.'gas'
    species2 (str): second particle species, i.e. 'star'
    zcut (float): value of the cut on |z|
    '''
    #gas
    angles = np.linspace(0, 360, 16, endpoint=False)
    theta = np.radians(angles)

    x_ = np.cos(theta)*8
    y_ = np.sin(theta)*8
    part_gas = gizmo.gizmo_io.Read.read_snapshots([species1], 'index', simnum, simulation_directory=simdir, assign_hosts_rotation=True, assign_hosts=True)
    data_vols = subselect_solar_cyls(simdir, simnum, 'star', Rcyl, numvols, zcut)
    colors = [cmr.infinity(i / len(data_vols['z'])) for i in range(len(data_vols['z']))]
    x_gas      = part_gas[species1].prop('host.distance.principal.cartesian')[:,0]
    y_gas      = part_gas[species1].prop('host.distance.principal.cartesian')[:,1]
    z_gas      = part_gas[species1].prop('host.distance.principal.cartesian')[:,2]
    temp_gas   = part_gas[species1]['temperature']
    mass_gas   = part_gas[species1]['mass']

    cold_dense_gas_indices = np.where((temp_gas <= 10e4) & (part_gas[species1]['density'] > 10))
    x_cdg     = x_gas[cold_dense_gas_indices]
    y_cdg     = y_gas[cold_dense_gas_indices]
    z_cdg     = z_gas[cold_dense_gas_indices]
    mass_cdg  = mass_gas[cold_dense_gas_indices]

    spatial_cut = np.where((np.abs(x_cdg) <= 15) & (np.abs(y_cdg) <= 15)& (np.abs(z_cdg) <= zcut))
    x_cdg_cut     = x_cdg[spatial_cut]
    y_cdg_cut     = y_cdg[spatial_cut]
    mass_cdg_cut  = mass_cdg[spatial_cut]

    x_range_cut = (np.min(x_cdg_cut), np.max(x_cdg_cut))
    y_range_cut = (np.min(y_cdg_cut), np.max(y_cdg_cut))
    num_bins_x = 300
    num_bins_y = 300

    bin_width = (x_range_cut[1] - x_range_cut[0]) / num_bins_x
    bin_height = (y_range_cut[1] - y_range_cut[0]) / num_bins_y

    surface_density = mass_cdg_cut / (bin_width * bin_height)

    #stars
    part_star = gizmo.gizmo_io.Read.read_snapshots([species2], 'index', simnum, simulation_directory=simdir, assign_hosts_rotation=True, assign_hosts=True)
    x  = part_star[species2].prop('host.distance.principal.cartesian')[:,0]
    y  = part_star[species2].prop('host.distance.principal.cartesian')[:,1]
    z  = part_star[species2].prop('host.distance.principal.cartesian')[:,2]
    mass = mass = part_star[species2].prop('mass')
    disk_indices = np.where((np.abs(x) < 15) & (np.abs(y) < 15) & (np.abs(z) < zcut))
    x_masked     = x[disk_indices]
    y_masked     = y[disk_indices]
    mass_masked  = mass[disk_indices]

    ms,  xes,  yes,  bns  = stats.binned_statistic_2d(x_masked,  y_masked,  mass_masked,  'sum', range=[[-15, 15], [-15, 15]], bins=[300,300])
    x_range_cut = (np.min(x_masked), np.max(x_masked))
    dx = np.diff(xes)[:]  
    dy = np.diff(yes)[:]  
    area_elements = dx[:] * dy[:] 

    area_matrix = np.tile(area_elements,(300,1))
    dens_s = ms.T/area_matrix

    plt.rcParams["figure.figsize"] = (16, 8)
    fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)

    # Plot 1: Gas
    axs[0].tick_params(top=True, right=True)
    axs[0].set_aspect('equal')
    h = axs[0].hist2d(x_cdg_cut, y_cdg_cut, weights=surface_density, bins=[num_bins_x, num_bins_y], norm=LogNorm(), cmap=cmr.sapphire)
    cbar = plt.colorbar(h[3], ax=axs[0],fraction=0.046, pad=0.04)
    cbar.set_label(r'$\Sigma_{cold \ gas}$ [M$_{\odot}$ kpc$^{-2}$]', fontsize=20, rotation=270, labelpad=30)
    cbar.ax.tick_params(labelsize=20)

    legend_handles = []
    for i in range(len(data_vols['x'])):
        center_x = float(x_[i])
        center_y = float(y_[i])
        circle = mpatches.Circle((center_x, center_y), radius=np.sqrt(2.4), edgecolor=colors[i], fill=False, linewidth=3.0)
        underlay = mpatches.Circle((center_x, center_y), radius=np.sqrt(2.4), edgecolor='black', fill=False, linewidth=5)
        axs[0].add_patch(underlay)
        axs[0].add_patch(circle)
        text = axs[0].text(center_x, center_y, f'{i + 1}', ha='center', va='center', color='white', fontsize=15, fontdict={'weight': 'bold'})
        text.set_path_effects([withStroke(linewidth=3, foreground='k')])
        legend_handles.append(mpatches.Patch(facecolor='none', edgecolor=colors[i], label=f'Circle {i+1}', linewidth=2))

    V = np.array([8,0])
    axs[0].quiver(0, 0, V[0], V[1], angles='xy', scale_units='xy', scale=1, color='k')
    vec_label = axs[0].text(0, 0.75, r'R$_{\mathrm{cyl}}$ = 8 kpc', color='white', alpha=1, fontsize= 13, fontdict={'weight': 'bold'})  
    vec_label.set_path_effects([withStroke(linewidth=3, foreground='k')])
    axs[0].legend(handles=legend_handles, labels=[fr'N$_*$={len(data_vols["z"][i])}' for i in range(len(data_vols['x']))], ncol=4, loc='upper center', framealpha=0.9)
    axs[0].set_xlabel('x [kpc]', fontsize=20)
    axs[0].set_ylabel('y [kpc]', fontsize=20)
    axs[0].tick_params(axis='both', which='major', labelsize=20)

    # Plot 2: Stars
    axs[1].tick_params(top=True, right=True)
    axs[1].set_aspect('equal')
    cs = axs[1].pcolormesh(xes, yes, np.log(dens_s), cmap=cmr.torch)
    cbar = plt.colorbar(cs, ax=axs[1],fraction=0.046, pad=0.04)
    cbar.set_label(r'$\Sigma_{*}$ [M$_{\odot}$ kpc$^{-2}$]', fontsize=20, rotation=270, labelpad=30)
    cbar.ax.tick_params(labelsize=20)

    for i in range(len(data_vols['x'])):
        center_x = float(x_[i])
        center_y = float(y_[i])
        circle = mpatches.Circle((center_x, center_y), radius=np.sqrt(2.4), edgecolor=colors[i], fill=False, linewidth=3.0)
        underlay = mpatches.Circle((center_x, center_y), radius=np.sqrt(2.4), edgecolor='black', fill=False, linewidth=5)
        axs[1].add_patch(underlay)
        axs[1].add_patch(circle)
        text = axs[1].text(center_x, center_y, f'{i + 1}', ha='center', va='center', color='white', fontsize=15, fontdict={'weight': 'bold'})
        text.set_path_effects([withStroke(linewidth=3, foreground='k')])

    axs[1].set_xlabel('x [kpc]', fontsize=20)
    axs[1].tick_params(axis='both', which='major', labelsize=20)
    axs[1].text(-4, -14, fr'm12i at z=0', color='white', alpha=1, fontsize= 20, fontdict={'weight': 'bold'})
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.gca().set_aspect('equal')
    plt.subplots_adjust(wspace=-0.25)
    plt.tight_layout()
    plt.show()

def generate_mean_stellar_motion_plot(simdir, simnum, species, Rcyl, numvols, zcut): #use zcut=1.5
    '''
    Generate 2-panel plot of mean stellar motion in the radial (R) and vertical (z) directions.
    simdir (str): filepath to directory where sim is located
    snapnum (int): snapshot number (e.g., 600)
    species (str): 'star', 'gas', 'dark' or 'all'
    zcut (float): value of the cut on |z|
    '''
    part_star = gizmo.gizmo_io.Read.read_snapshots([species], 'index', simnum, simulation_directory=simdir, assign_hosts_rotation=True, assign_hosts=True)
    x  = part_star[species].prop('host.distance.principal.cartesian')[:,0]
    y  = part_star[species].prop('host.distance.principal.cartesian')[:,1]
    z  = part_star[species].prop('host.distance.principal.cartesian')[:,2]
    Vrxy   = part_star['star'].prop('host.velocity.principal.cylindrical')[:,0]
    Vzxy   = part_star['star'].prop('host.velocity.principal.cylindrical')[:,2]
    angles = np.linspace(0, 360, 16, endpoint=False)
    theta = np.radians(angles)

    x_ = np.cos(theta)*8
    y_ = np.sin(theta)*8
    data_vols = subselect_solar_cyls(simdir, simnum, 'star', Rcyl, numvols, zcut)
    colors = [cmr.infinity(i / len(data_vols['z'])) for i in range(len(data_vols['z']))]
    plt.style.use('default')
    plt.rcParams["figure.figsize"] = (9, 12)


    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    #vz
    ax = axs[1]

    disk_indices = np.where((np.abs(x) < 15) & (np.abs(y) < 15) & (np.abs(z) < zcut))
    x_masked = x[disk_indices]
    y_masked = y[disk_indices]
    vz_masked = Vzxy[disk_indices]

    mean_vz, xe, ye, bn = scipy.stats.binned_statistic_2d(x_masked, y_masked, vz_masked, statistic='mean', range=[[-15, 15], [-15, 15]], bins=[300, 300])
    cs = ax.pcolormesh(xe, ye, mean_vz.T, cmap='RdBu', vmin=-35, vmax=35) 
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label(r"$\langle$V$_{z}$$\rangle$ [km s$^{-1}$]", fontsize=25, rotation=270, labelpad=30)
    cbar.ax.tick_params(labelsize=25)

    for i in range(len(data_vols['x'])):
        center_x = float(x_[i])
        center_y = float(y_[i])
        circle = mpatches.Circle((center_x, center_y), radius=np.sqrt(2.4), edgecolor=colors[i], fill=False, linewidth=3.0)
        underlay = mpatches.Circle((center_x, center_y), radius=np.sqrt(2.4), edgecolor='black', fill=False, linewidth=4.5)
        ax.add_patch(underlay)
        ax.add_patch(circle)
        text = ax.text(center_x, center_y, f'{i + 1}', ha='center', va='center', color='white', fontsize=15, fontdict={'weight': 'bold'})
        text.set_path_effects([withStroke(linewidth=3, foreground='k')])
    ax.tick_params(top=False, right=False)
    ax.tick_params(axis='both', which='major', labelsize=25, width=2, length=10)  
    ax.set_xlabel('x [kpc]', fontsize=25)
    ax.set_ylabel('y [kpc]', fontsize=25)

    # vR
    ax = axs[0]

    disk_indices = np.where((np.abs(x) < 15) & (np.abs(y) < 15) & (np.abs(z) < zcut))
    x_masked = x[disk_indices]
    y_masked = y[disk_indices]
    vz_masked = Vzxy[disk_indices]
    vr_masked = Vrxy[disk_indices]

    mean_vr, xe, ye, bn = scipy.stats.binned_statistic_2d(x_masked, y_masked, vr_masked, statistic='mean', range=[[-15, 15], [-15, 15]], bins=[300, 300])
    cs = ax.pcolormesh(xe, ye, mean_vr.T, cmap='RdBu', vmin=-75, vmax=75) #80
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label(r"$\langle$V$_{R}$$\rangle$ [km s$^{-1}$]", fontsize=25, rotation=270, labelpad=30)
    cbar.ax.tick_params(labelsize=20)

    for i in range(len(data_vols['x'])):
        center_x = float(x_[i])
        center_y = float(y_[i])
        circle = mpatches.Circle((center_x, center_y), radius=np.sqrt(2.4), edgecolor=colors[i], fill=False, linewidth=3.0)
        underlay = mpatches.Circle((center_x, center_y), radius=np.sqrt(2.4), edgecolor='black', fill=False, linewidth=4.5)
        ax.add_patch(underlay)
        ax.add_patch(circle)
        text = ax.text(center_x, center_y, f'{i + 1}', ha='center', va='center', color='white', fontsize=15, fontdict={'weight': 'bold'})
        text.set_path_effects([withStroke(linewidth=3, foreground='k')])
    cbar.ax.tick_params(labelsize=25)
    ax.tick_params(top=False, right=False)
    ax.tick_params(axis='both', which='major', labelsize=25, width=2, length=10)  
    ax.set_xlabel('', fontsize=25)
    ax.set_ylabel('y [kpc]', fontsize=25)  
    plt.subplots_adjust(hspace=0.05)
    plt.tight_layout()
    plt.show()

def generate_gal_cyl_feh_mgfe_plot(simdir, simnum, species, Rcyl, numvols, zcut): #use zcut=1.5
    '''
    Generate 2-panel plot of mean [Fe/H] and [Mg/Fe] in the Galactic cylinder.
    simdir (str): filepath to directory where sim is located
    snapnum (int): snapshot number (e.g., 600)
    species (str): 'star', 'gas', 'dark' or 'all'
    zcut (float): value of the cut on |z|
    '''
    part_star = gizmo.gizmo_io.Read.read_snapshots([species], 'index', simnum, simulation_directory=simdir, assign_hosts_rotation=True, assign_hosts=True)
    x  = part_star[species].prop('host.distance.principal.cartesian')[:,0]
    y  = part_star[species].prop('host.distance.principal.cartesian')[:,1]
    z  = part_star[species].prop('host.distance.principal.cartesian')[:,2]
    age  = part_star[species].prop('age')
    feh  = part_star[species].prop('metallicity.fe')
    mgfe = part_star[species].prop('metallicity.mg - metallicity.fe')
    angles = np.linspace(0, 360, 16, endpoint=False)
    theta = np.radians(angles)

    x_ = np.cos(theta)*8
    y_ = np.sin(theta)*8
    data_vols = subselect_solar_cyls(simdir, simnum, 'star', Rcyl, numvols, zcut)
    colors = [cmr.infinity(i / len(data_vols['z'])) for i in range(len(data_vols['z']))]
    plt.style.use('default')
    plt.rcParams["figure.figsize"] = (9, 12)


    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    # MgFe
    ax = axs[1]

    disk_indices = np.where((np.abs(x) < 15) & (np.abs(y) < 15) & (np.abs(z) < zcut) & (age <= 2.5))
    x_masked     = x[disk_indices]
    y_masked     = y[disk_indices]
    feh_masked    = feh[disk_indices]
    mgfe_masked    = mgfe[disk_indices]

    num_bins_x = 300
    num_bins_y = 300
    mean_mgfe, xe, ye, bn = scipy.stats.binned_statistic_2d(x_masked, y_masked, mgfe_masked, statistic='mean', range=[[-15, 15], [-15, 15]], bins=[num_bins_x,num_bins_y])
    cs = ax.pcolormesh(xe, ye, mean_mgfe.T, cmap='RdBu_r', vmin=0.14, vmax=0.3) #0.22 to 0.28
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label(r"$\langle$[Mg/Fe]$\rangle$ [dex]", fontsize=25, rotation=270, labelpad=30)
    cbar.ax.tick_params(labelsize=25)
    cbar.formatter = ticker.FormatStrFormatter('%.2f')  # Set to 2 significant figures
    cbar.update_ticks()
    for i in range(len(data_vols['x'])):
        center_x = float(x_[i])
        center_y = float(y_[i])
        circle = mpatches.Circle((center_x, center_y), radius=np.sqrt(2.4), edgecolor=colors[i], fill=False, linewidth=3.0)
        underlay = mpatches.Circle((center_x, center_y), radius=np.sqrt(2.4), edgecolor='black', fill=False, linewidth=4.5)
        ax.add_patch(underlay)
        ax.add_patch(circle)
        text = ax.text(center_x, center_y, f'{i + 1}', ha='center', va='center', color='white', fontsize= 15, fontdict={'weight': 'bold'})
        text.set_path_effects([withStroke(linewidth=3, foreground='k')])
    ax.tick_params(top=False, right=False, labelsize=25)
    ax.tick_params(axis='both', which='major', labelsize=25, width=2, length=10)
    ax.set_xlabel('x [kpc]', fontsize=25)
    ax.set_ylabel('y [kpc]', fontsize=25)
    # FeH
    ax = axs[0]
    mean_feh, xe, ye, bn = scipy.stats.binned_statistic_2d(x_masked, y_masked, feh_masked, statistic='mean', range=[[-15, 15], [-15, 15]], bins=[num_bins_x,num_bins_y])
    cs = ax.pcolormesh(xe, ye, mean_feh.T, cmap='RdBu_r', vmin=-0.5, vmax=0.3) #10.9 to 0.0
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label(r"$\langle$[Fe/H]$\rangle$ [dex]", fontsize=25, rotation=270, labelpad=30)    
    cbar.ax.tick_params(labelsize=25)
    cbar.formatter = ticker.FormatStrFormatter('%.2f')  # Set to 2 significant figures
    cbar.update_ticks()
    for i in range(len(data_vols['x'])):
        center_x = float(x_[i])
        center_y = float(y_[i])
        circle = mpatches.Circle((center_x, center_y), radius=np.sqrt(2.4), edgecolor=colors[i], fill=False, linewidth=3.0)
        underlay = mpatches.Circle((center_x, center_y), radius=np.sqrt(2.4), edgecolor='black', fill=False, linewidth=4.5)
        ax.add_patch(underlay)
        ax.add_patch(circle)
        text = ax.text(center_x, center_y, f'{i + 1}', ha='center', va='center', color='white', fontsize= 15, fontdict={'weight': 'bold'})
        text.set_path_effects([withStroke(linewidth=3, foreground='k')])
    cbar.ax.tick_params(labelsize=25)
    ax.tick_params(top=False, right=False, labelsize=25)
    ax.tick_params(axis='both', which='major', labelsize=25, width=2, length=10)
    ax.set_xlabel('', fontsize=25)
    ax.set_ylabel('y [kpc]', fontsize=25)  
    plt.subplots_adjust(hspace=0.05)
    plt.tight_layout()
    plt.show()

def generate_vertical_feh_mgfe_profile_plot(simdir, simnum, species, Rcyl, numvols, zcut): #use zcut=10
    '''
    Generate 2-panel plot of vertical metallicity profiles for [Fe/H] and [Mg/Fe] (and a comparison with corresponding [Fe/H] plot from Graf et al. 2024).
    simdir (str): filepath to directory where sim is located
    snapnum (int): snapshot number (e.g., 600)
    species (str): 'star', 'gas', 'dark' or 'all'
    zcut (float): value of the cut on |z|
    '''
    data_vols = subselect_solar_cyls(simdir, simnum, species, Rcyl, numvols, zcut)
    angles = np.linspace(0, 360, 16, endpoint=False)
    theta = np.radians(angles)
    x_ = np.cos(theta)*8
    y_ = np.sin(theta)*8
    # Masked for low vz (-10 to 10 km/s)
    data_keys = ['vz', 'z', 'feh', 'mgfe']
    masked_data = {key: [] for key in data_keys}

    for i in range(len(data_vols['x'])):
        mask_low_vz = np.where(np.abs(data_vols['vz'][i]) <= 10)
        mask_low_vz_indices = mask_low_vz[0]
        
        for key in data_keys:
            data_from_dict = data_vols[key]
            masked_data[key].append(data_from_dict[i][mask_low_vz_indices])

    def best_fit_slope_and_intercept(xs, ys):
        xy_mean = mean(x * y for x, y in zip(xs, ys)) #
        x_squared_mean = mean(x**2 for x in xs)
        x_mean = mean(xs)
        y_mean = mean(ys)
        m = (xy_mean - x_mean * y_mean) / (x_squared_mean - x_mean**2) #cov(x,y)/var(x)
        b = y_mean - m * x_mean
        return m, b
    slopes_abs_z = {key: [] for key in data_keys}
    int_abs_z = {key: [] for key in data_keys}

    for i in range(len(data_vols['x'])):
        for key in data_keys:
            x_values = masked_data['z'][i]
            y_values = masked_data[key][i]
            
            # using absolute values of z
            abs_x_values = [abs(x) for x in x_values]
            corresponding_y_values = y_values  # y_values remain unchanged

            slope, intercept = best_fit_slope_and_intercept(abs_x_values, corresponding_y_values) 
            slopes_abs_z[key].append(slope)
            int_abs_z[key].append(intercept)

    best_fit_abs = {key: [] for key in data_keys}
    z_abs = []

    for i in range(len(data_vols['x'])):
        for key in data_keys:
            x_values = masked_data['z'][i]
            y_values = masked_data[key][i]
            
            # using absolute values of z
            abs_x_values = [abs(x) for x in x_values]
            bf = (slopes_abs_z[key][i] * np.array(abs_x_values)) + int_abs_z[key][i]
            best_fit_abs[key].append(bf)
        z_abs.append(np.array(abs_x_values))
    mean_m_feh = np.mean(slopes_abs_z['feh'])
    mean_b_feh = np.mean(int_abs_z['feh'])
    median_m_feh = np.median(slopes_abs_z['feh'])
    median_b_feh = np.median(int_abs_z['feh'])

    mean_m_mgfe = np.mean(slopes_abs_z['mgfe'])
    mean_b_mgfe = np.mean(int_abs_z['mgfe'])
    median_m_mgfe = np.median(slopes_abs_z['mgfe'])
    median_b_mgfe = np.median(int_abs_z['mgfe'])
    print(mean_m_feh, mean_b_feh, np.std(slopes_abs_z['feh']))
    print(mean_m_mgfe, mean_b_mgfe, np.std(slopes_abs_z['mgfe']))
    bf_mean = (mean_m_feh*np.array(z_abs[0]))+mean_b_feh
    bf_median = (median_m_feh*np.array(z_abs[0]))+median_b_feh
    ub_mean = (mean_m_feh+np.std(slopes_abs_z['feh']))*z_abs[0]+(mean_b_feh+np.std(int_abs_z['feh']))
    lb_mean = (mean_m_feh-np.std(slopes_abs_z['feh']))*z_abs[0]+(mean_b_feh-np.std(int_abs_z['feh']))
    ub_median = (median_m_feh+np.std(slopes_abs_z['feh']))*z_abs[0]+(median_b_feh+np.std(int_abs_z['feh']))
    lb_median = (median_m_feh-np.std(slopes_abs_z['feh']))*z_abs[0]+(median_b_feh-np.std(int_abs_z['feh']))
    bf_mean_mgfe = (mean_m_mgfe*np.array(z_abs[0]))+mean_b_mgfe
    bf_median_mgfe = (median_m_mgfe*np.array(z_abs[0]))+median_b_mgfe
    column_names = ['z', 'feh']
    graf_avg = pd.read_csv('https://drive.google.com/file/d/1yG98Un6vK9at9-I6oEs98idzAiBxtf2m/view?usp=drive_link', names=column_names, header=None, delimiter='\t') 
    graf_ub = pd.read_csv('https://drive.google.com/file/d/1yc84PWXCw-IRF3Mv6ZRjIKUI2flzmgoe/view?usp=drive_link', names=column_names, header=None, delimiter='\t') 
    graf_lb = pd.read_csv('https://drive.google.com/file/d/1HyFKRguGFGFt2vf0p1oENHcI02fkExpA/view?usp=drive_link', names=column_names, header=None, delimiter='\t') 
    ub = ((mean_m_feh+np.std(slopes_abs_z['feh']))*np.array(z_abs[0]))+(mean_b_feh+np.std(int_abs_z['feh']))
    lb = ((mean_m_feh-np.std(slopes_abs_z['feh']))*np.array(z_abs[0]))+(mean_b_feh-np.std(int_abs_z['feh']))
    ub3 = ((mean_m_feh+3*np.std(slopes_abs_z['feh']))*np.array(z_abs[0]))+(mean_b_feh+3*np.std(int_abs_z['feh']))
    lb3 = ((mean_m_feh-3*np.std(slopes_abs_z['feh']))*np.array(z_abs[0]))+(mean_b_feh-3*np.std(int_abs_z['feh']))
    ub_mgfe = ((mean_m_mgfe+np.std(slopes_abs_z['mgfe']))*np.array(z_abs[0]))+(mean_b_mgfe+np.std(int_abs_z['mgfe']))
    lb_mgfe = ((mean_m_mgfe-np.std(slopes_abs_z['mgfe']))*np.array(z_abs[0]))+(mean_b_mgfe-np.std(int_abs_z['mgfe']))
    ub3_mgfe = ((mean_m_mgfe+3*np.std(slopes_abs_z['mgfe']))*np.array(z_abs[0]))+(mean_b_mgfe+3*np.std(int_abs_z['mgfe']))
    lb3_mgfe = ((mean_m_mgfe-3*np.std(slopes_abs_z['mgfe']))*np.array(z_abs[0]))+(mean_b_mgfe-3*np.std(int_abs_z['mgfe']))
    plt.style.use('default')
    plt.rcParams["figure.figsize"] = (9, 12)


    fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)
    colors = [cmr.infinity(i / 15) for i in range(16)]
    # MgFe
    ax = axs[1]
    ax.plot(z_abs[0], bf_mean_mgfe, c='black', ls='-', linewidth=2.5, label='m12i mean best-fit', zorder=10)
    ax.fill_between(z_abs[0][3884:5778], ub3_mgfe[3884:5778], lb3_mgfe[3884:5778], color='black', alpha=0.4, label=r'3-$\sigma$', interpolate = True)
    ax.set_ylim(0.1, 0.35)
    ax.set_xlim(0, 1.5)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_xlabel(fr"$|z|$ [{u.kpc:latex_inline}]", ha='center', va='center', fontsize=25, labelpad=30)
    ax.set_ylabel(r"[Mg/Fe] [dex]", ha='center', va='center', rotation='vertical', fontsize=25, labelpad=30)
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.legend(fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=25, width=2, length=10)
    ax.text(0.05, 0.13, r'$\frac{d[\mathrm{Mg/Fe}]}{d\mathrm{z}}=0.014\ [\mathrm{dex}\ \mathrm{kpc}^{-1}]$', c='black',fontsize=25)

    #FeH
    ax = axs[0]
    ax.plot(z_abs[0], bf_mean, c='black', ls='-', linewidth=2.5, label='m12i mean best-fit', zorder=10)
    ax.fill_between(z_abs[0][3884:5778], ub3[3884:5778], lb3[3884:5778], color='black', alpha=0.4, label=r'3-$\sigma$', interpolate = True)
    ax.plot(graf_avg['z'], graf_avg['feh'], color=colors[15], linestyle='--', label='Mean (Graf et al. 2024)', alpha=0.75, zorder=6)
    ax.fill_between(graf_ub['z'], graf_lb['feh'], graf_ub['feh'], color=colors[15], alpha=0.1, label=r'1-$\sigma$', zorder=5)
    ax.set_ylim(-0.45, 0.2)
    ax.set_xlim(0, 1.5)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_ylabel(r"[Fe/H] [dex]", ha='center', va='center', rotation='vertical', fontsize=25, labelpad=30)
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.tick_params(axis='both', which='major', labelsize=25, width=2, length=10)
    ax.legend(fontsize=18)
    ax.text(0.05, -0.36, r'$\frac{d[\mathrm{Fe/H}]}{d\mathrm{z}}=-0.129\ [\mathrm{dex}\ \mathrm{kpc}^{-1}]$', c='black', fontsize=25)
    plt.subplots_adjust(hspace=0.05)
    plt.tight_layout()
    plt.show()

def generate_azim_avgd_met_grad_plot(simdir, simnum, species, Rcyl, numvols, zcut): #use zcut=10
    '''
    Generate 2-panel plot of azimuthally averaged metallicity gradient for [Fe/H] and [Mg/Fe].
    simdir (str): filepath to directory where sim is located
    snapnum (int): snapshot number (e.g., 600)
    species (str): 'star', 'gas', 'dark' or 'all'
    zcut (float): value of the cut on |z|
    '''
    data_vols = subselect_solar_cyls(simdir, simnum, species, Rcyl, numvols, zcut)
    z_array_list = []
    vz_array_list = []
    max_z_list = []
    bdata_list_mgfe = []

    for i in range(len(data_vols['x'])):
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
            label=data_vols['mgfe'][i],
            bins=zvz_bins,
            units=galactic,
            s_N_thresh=32,
        )
        bdata_list_mgfe.append(bdata)
    bdata_list_feh = []

    for i in range(len(data_vols['x'])):
        z_array = data_vols['z'][i] * u.kpc
        vz_array = data_vols['vz'][i] * (u.km / u.s)
        max_z = np.round(3 * 1.5 * MAD(z_array), 1)
        max_vz = np.round(3 * 1.5 * MAD(vz_array), 0)
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
        bdata_list_feh.append(bdata)
    plt.style.use('default')
    plt.rcParams["figure.figsize"] = (9, 12)


    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    #FeH
    ax = axs[0]
    sum_vz = np.zeros((100, 100))
    sum_z = np.zeros((100, 100))
    sum_feh = np.zeros((100, 100))

    for i in range(16):
        sum_vz += bdata_list_feh[i]["vel"].to_value(u.km/u.s)
        sum_z += bdata_list_feh[i]["pos"].to_value(u.kpc)
        sum_feh += bdata_list_feh[i]["label"]

    mean_vz = sum_vz / 16
    mean_z = sum_z / 16
    mean_feh = sum_feh / 16

    X, Y = np.meshgrid(mean_vz[:,0], mean_z[0,:])
    cs = ax.pcolormesh(X, Y, mean_feh, cmap=cmr.torch)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlim(-150, 150)

    cbar = plt.colorbar(cs, ax=ax,fraction=0.043, pad=0.04)
    cbar.set_label(r"$\langle$[Fe/H]$\rangle$ [dex]", fontsize=20, rotation=270, labelpad=30)
    cbar.ax.tick_params(labelsize=20)
    ax.set_xlabel(f"", fontsize=20)
    ax.set_ylabel(f"$z$ [{u.kpc:latex_inline}]", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_aspect(40)


    # MgFe
    ax = axs[1]
    sum_vz = np.zeros((100, 100))
    sum_z = np.zeros((100, 100))
    sum_mgfe = np.zeros((100, 100))

    for i in range(16):
        sum_vz += bdata_list_mgfe[i]["vel"].to_value(u.km/u.s)
        sum_z += bdata_list_mgfe[i]["pos"].to_value(u.kpc)
        sum_mgfe += bdata_list_mgfe[i]["label"]

    mean_vz = sum_vz / 16
    mean_z = sum_z / 16
    mean_mgfe = sum_mgfe / 16

    X, Y = np.meshgrid(mean_vz[:,0], mean_z[0,:])
    cs = ax.pcolormesh(X, Y, mean_mgfe, vmin=0.18, vmax=0.35, cmap=cmr.torch)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlim(-100, 100)

    cbar = plt.colorbar(cs, ax=ax,fraction=0.043, pad=0.04)
    cbar.set_label(r"$\langle$[Mg/Fe]$\rangle$ [dex]", fontsize=20, rotation=270, labelpad=30)
    cbar.ax.tick_params(labelsize=20)
    ax.set_xlabel(f"$v_z$ [{u.km/u.s:latex_inline}]", fontsize=20)
    ax.set_ylabel(f"$z$ [{u.kpc:latex_inline}]", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_aspect(40)
    plt.subplots_adjust(wspace=-0.4)
    plt.tight_layout()
    plt.show()

def generate_data_model_residual_plot(simdir, simnum, species, Rcyl, numvols, zcut, idx):
    '''
    Generate 3-panel plot of FIRE data, OTI best-fit model, and normalized residuals.
    simdir (str): filepath to directory where sim is located
    snapnum (int): snapshot number (e.g., 600)
    species (str): 'star', 'gas', 'dark' or 'all'
    zcut (float): value of the cut on |z|
    idx (int): index of volume to plot
    '''
    res_list, bdata_list, model_list = run_oti_analysis(simdir, simnum, species, Rcyl, numvols, zcut)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharex=True, sharey=True, constrained_layout=True)
        
    cs = axes[0].pcolormesh(
    bdata_list[idx]["vel"].to_value(u.km/u.s),
    bdata_list[idx]["pos"].to_value(u.kpc),
    bdata_list[idx]["label"],
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
    
    model_feh = model_list[idx].get_label(bdata_list[idx]["pos"], bdata_list[idx]["vel"], res_list[idx].params)
    cs = axes[1].pcolormesh(
        bdata_list[idx]["vel"].to_value(u.km / u.s),
        bdata_list[idx]["pos"].to_value(u.kpc),
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
        bdata_list[idx]["vel"].to_value(u.km / u.s),
        bdata_list[idx]["pos"].to_value(u.kpc),
        (bdata_list[idx]["label"] - model_feh) / bdata_list[idx]["label_err"],
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
        
    axes[0].set_title(f"FIRE Data V{idx+1}", fontsize=25)
    axes[1].set_title("OTI Fitted Model", fontsize=25)
    axes[2].set_title("Normalized Residuals", fontsize=25)
    plt.show() 

    