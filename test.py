from load_data import load_simulation_data
from subselect_data import subselect_solar_cyls
from oti_analysis import *
from plot_data import *
simpath = '/Users/micahoeur/Dropbox/Research/Sarah/FIRE2_m12i_metal_diffusion/output_with_accel'
snum = 600
spcs = 'star'
minval_feh = -0.5 
maxval_feh = 0.3 
#load_simulation_data(simpath, snum, spcs)
#print("load_simulation_data ran successfully")
#subselect_solar_cyls(simpath, snum, spcs, 8, 16, 10)
#print("subselect_solar_cyls ran successfully")
#run_oti_analysis(simpath, snum, spcs, 8, 16, 10)
#print("run_oti_analysis ran successfully")
#plot_oti_results(simpath, snum, spcs, 8, 16, 10, 0)
#print("plot_oti_results ran successfully")
#generate_surface_mass_density_plot(simpath, snum, 'gas', 'star', 8, 16, 1.5)
#print("ge#nerate_surface_mass_density_plot ran successfully")
#generate_mean_stellar_motion_plot(simpath, snum, spcs, 8, 16, 1.5)
#print("generate_mean_stellar_motion_plot ran successfully")
#generate_gal_cyl_feh_mgfe_plot(simpath, snum, spcs, 8, 16, 1.5,  minval_mgfe, maxval_mgfe, minval_feh, maxval_feh)
#print("generate_gal_cyl_feh_mgfe_plot ran successfully")
#generate_vertical_feh_mgfe_profile_plot(simpath, snum, spcs, 8, 16, 10)
#print("generate_vertical_feh_mgfe_profile_plot ran successfully")
#generate_azim_avgd_met_grad_plot(simpath, snum, spcs, 8, 16, 10, minval_mgfe, maxval_mgfe)
#print("generate_azim_avgd_met_grad_plot ran successfully")
#generate_data_model_residual_plot(simpath, snum, spcs, 8, 16, 10, minval_feh, maxval_feh)
#print("generate_data_model_residual_plot ran successfully")
#generate_vertical_acceleration_profiles_plot(simpath, snum, spcs, 8, 16, 10)
#print("generate_vertical_acceleration_profiles_plot ran successfully")
generate_normalized_residuals_plot(simpath, snum, spcs, 8, 16, 10)
print("generate_normalized_residuals_plot ran successfully")
#generate_stellar_smd_plot(simpath, snum, spcs, 8, 16, 10, minval_feh, maxval_feh)
#print("generate_stellar_smd_plot ran successfully")
#generate_metallicity_gradient_plot(simpath, snum, spcs, 8, 16, 10, 'feh', 'Fe/H', -1.2, 0.1)
#print("generate_metallicity_gradient_plot ran successfully")
#generate_stargasdm_rho_plot(simpath, snum, 'all', 8, 16, 5)
#print("generate_stargasdm_rho_plot ran successfully")
#generate_asymmetry_figofmer(simpath, snum, 'all', 8, 16, 5, [1, 11, 13, 15])
#print("generate_asymmetry_figofmer ran successfully")
#print("all functions ran successfully")