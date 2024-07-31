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
from matplotlib.patheffects import withStroke
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'font.size': 20})

plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

def generate_surface_mass_density_plot(simdir, simnum, species1, species2, zcut): #use zcut=1.5
    #res_list = run_oti_analysis(simdir, simnum, species, zcut)
    #gas
    angles = np.linspace(0, 360, 16, endpoint=False)
    theta = np.radians(angles)

    x_ = np.cos(theta)*8
    y_ = np.sin(theta)*8
    part_gas = gizmo.gizmo_io.Read.read_snapshots([species1], 'index', simnum, simulation_directory=simdir, assign_hosts_rotation=True, assign_hosts=True)
    data_vols = subselect_solar_cyls(simdir, simnum, 'star', zcut)
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

def generate_mean_stellar_motion_plot(simdir, simnum, species, zcut): #use zcut=1.5
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
    data_vols = subselect_solar_cyls(simdir, simnum, 'star', zcut)
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

def generate_gal_cyl_feh_mgfe_plot(simdir, simnum, species, zcut): #use zcut=1.5
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
    data_vols = subselect_solar_cyls(simdir, simnum, 'star', zcut)
    colors = [cmr.infinity(i / len(data_vols['z'])) for i in range(len(data_vols['z']))]
    plt.style.use('default')
    plt.rcParams["figure.figsize"] = (9, 12) #previously (20,8) with 1 row 2 cols


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

    for i in range(len(data_vols['x'])):
        center_x = float(x_[i])
        center_y = float(y_[i])
        #plt.scatter(data_vols['x'][i], data_vols['y'][i], c=colors[i], alpha=0.1)
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