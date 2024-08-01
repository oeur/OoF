from load_data import load_simulation_data
from subselect_data import subselect_solar_cyls
from oti_analysis import run_oti_analysis
from oti_analysis import plot_oti_results
from plot_data import generate_surface_mass_density_plot
from plot_data import generate_mean_stellar_motion_plot
from plot_data import generate_gal_cyl_feh_mgfe_plot
from plot_data import generate_vertical_feh_mgfe_profile_plot
from plot_data import generate_azim_avgd_met_grad_plot
from plot_data import generate_data_model_residual_plot
from plot_data import generate_vertical_acceleration_profiles_plot
simpath = '/Users/micahoeur/Dropbox/Research/Sarah/FIRE2_m12i_metal_diffusion/output_with_accel'
snum = 600
spcs = 'star'
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
#generate_gal_cyl_feh_mgfe_plot(simpath, snum, spcs, 8, 16, 1.5)
#print("generate_gal_cyl_feh_mgfe_plot ran successfully")
#generate_vertical_feh_mgfe_profile_plot(simpath, snum, spcs, 8, 16, 10)
#print("generate_vertical_feh_mgfe_profile_plot ran successfully")
#generate_azim_avgd_met_grad_plot(simpath, snum, spcs, 8, 16, 10)
#print("generate_azim_avgd_met_grad_plot ran successfully")
#generate_data_model_residual_plot(simpath, snum, spcs, 8, 16, 10, 0)
#print("generate_data_model_residual_plot ran successfully")
generate_vertical_acceleration_profiles_plot(simpath, snum, spcs, 8, 16, 10)
print("generate_vertical_acceleration_profiles_plot ran successfully")
#print("all functions ran successfully")