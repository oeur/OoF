import gizmo_analysis as gizmo

def load_simulation_data(simdir, snapnum, species):
    '''
    Load in the FIRE simulation data and read in specific arrays of data.
    simdir (str): filepath to directory where sim is located
    snapnum (int): snapshot number (e.g., 600)
    species (str): 'star', 'gas', 'dark' or 'all'
    '''
    part = gizmo.gizmo_io.Read.read_snapshots([species], 'index', snapnum, simulation_directory=simdir, assign_hosts_rotation=True, assign_hosts=True)
    part[species].prop('host.distance.principal.cylindrical')
    rxyz   = part[species].prop('host.distance.total')
    Rxy    = part[species].prop('host.distance.principal.cylindrical')[:,0]
    Phixy  = part[species].prop('host.distance.principal.cylindrical')[:,1]
    Zxy    = part[species].prop('host.distance.principal.cylindrical')[:,2]
    Vrxy   = part[species].prop('host.velocity.principal.cylindrical')[:,0]
    Vphixy = part[species].prop('host.velocity.principal.cylindrical')[:,1]
    Vzxy   = part[species].prop('host.velocity.principal.cylindrical')[:,2]

    x  = part[species].prop('host.distance.principal.cartesian')[:,0]
    y  = part[species].prop('host.distance.principal.cartesian')[:,1]
    z  = part[species].prop('host.distance.principal.cartesian')[:,2]
    vx = part[species].prop('host.velocity.principal.cartesian')[:,0]
    vy = part[species].prop('host.velocity.principal.cartesian')[:,1]
    vz = part[species].prop('host.velocity.principal.cartesian')[:,2]
    
    try:
        ax  = part[species].prop('host.acceleration.principal.cartesian')[:,0]
        ay  = part[species].prop('host.acceleration.principal.cartesian')[:,1]
        az  = part[species].prop('host.acceleration.principal.cartesian')[:,2]
    except KeyError:
        ax = ay = az = None 

    age  = part[species].prop('age')
    mass = part[species].prop('mass')
    feh  = part[species].prop('metallicity.fe')
    mgfe = part[species].prop('metallicity.mg - metallicity.fe')

    pos_stars = part[species].prop("host.distance.principal") #stellar positions with respect to the host. 
    pos = part[species].prop('host.distance.principal.cylindrical')
    dist_stars = part[species].prop("host.distance.total") #stellar distance with respect to the host.
    vel_stars = part[species].prop("host.velocity.principal")
    vels = part[species].prop('host.velocity.principal.cylindrical')
    return rxyz, Rxy, Phixy, Zxy, Vrxy, Vphixy, Vzxy, x, y, z, vx, vy, vz, ax, ay, az, age, mass, feh, mgfe, pos_stars, pos, dist_stars, vel_stars, vels
