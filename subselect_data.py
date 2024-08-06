from load_data import load_simulation_data
import numpy as np

def subselect_solar_cyls(simdir, simnum, species, Rcyl, numvols, zcut):
    # Unpack all returned values
    (rxyz, Rxy, Phixy, Zxy, Vrxy, Vphixy, Vzxy, x, y, z, 
     vx, vy, vz, ax, ay, az, age, mass, feh, mgfe, 
     pos_stars, pos, dist_stars, vel_stars, vels) = load_simulation_data(simdir, simnum, species)
    # Subselect the data
    angles = np.linspace(0, 360, numvols, endpoint=False)
    theta = np.radians(angles)

    x_ = np.cos(theta)*Rcyl
    y_ = np.sin(theta)*Rcyl
    keep_volumes = []
    for i in range(len(x_)):
        keep = np.where((((x - float(x_[i])) ** 2 + (y - float(y_[i])) ** 2 ) < 2) & (np.abs(z) < zcut))
        keep_volumes.append(keep)

    data_keys = ['vz', 'z', 'feh', 'mgfe','pos', 'vels', 'x', 'y', 'age', 'Vphixy', 'vx', 'vy', 'az', 'rxyz', 'mass', 'Vzxy', 'Vrxy']
    data_vols = {key: [] for key in data_keys}
    for keep in keep_volumes:
        for key in data_keys:
            # Check if the value for the key is None
            if locals()[key] is not None:
                data_vols[key].append(locals()[key][keep])
            else:
                data_vols[key].append(None)  # or handle as needed
    
    return data_vols