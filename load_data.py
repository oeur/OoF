import gizmo_analysis as gizmo

def load_simulation_data(simdir, snapnum, species):
    '''
    Load in the FIRE simulation data and read in specific arrays of data.
    simdir (str): filepath to directory where sim is located
    snapnum (int): snapshot number (e.g., 600)
    species (str): 'star', 'gas', 'dark' or 'all'
    '''
    part = gizmo.gizmo_io.Read.read_snapshots([species], 'index', snapnum, simulation_directory=simdir, assign_hosts_rotation=True, assign_hosts=True)
    part['star'].prop('host.distance.principal.cylindrical')
    rxyz   = part['star'].prop('host.distance.total')
    Rxy    = part['star'].prop('host.distance.principal.cylindrical')[:,0]
    Phixy  = part['star'].prop('host.distance.principal.cylindrical')[:,1]
    Zxy    = part['star'].prop('host.distance.principal.cylindrical')[:,2]
    Vrxy   = part['star'].prop('host.velocity.principal.cylindrical')[:,0]
    Vphixy = part['star'].prop('host.velocity.principal.cylindrical')[:,1]
    Vzxy   = part['star'].prop('host.velocity.principal.cylindrical')[:,2]

    x  = part['star'].prop('host.distance.principal.cartesian')[:,0]
    y  = part['star'].prop('host.distance.principal.cartesian')[:,1]
    z  = part['star'].prop('host.distance.principal.cartesian')[:,2]
    vx = part['star'].prop('host.velocity.principal.cartesian')[:,0]
    vy = part['star'].prop('host.velocity.principal.cartesian')[:,1]
    vz = part['star'].prop('host.velocity.principal.cartesian')[:,2]
    ax  = part['star'].prop('host.acceleration.principal.cartesian')[:,0]
    ay  = part['star'].prop('host.acceleration.principal.cartesian')[:,1]
    az  = part['star'].prop('host.acceleration.principal.cartesian')[:,2]

    age  = part['star'].prop('age')
    mass = part['star'].prop('mass')
    feh  = part['star'].prop('metallicity.fe')
    mgfe = part['star'].prop('metallicity.mg - metallicity.fe')

    pos_stars = part['star'].prop("host.distance.principal") #stellar positions with respect to the host. 
    pos = part['star'].prop('host.distance.principal.cylindrical')
    dist_stars = part['star'].prop("host.distance.total") #stellar distance with respect to the host.
    vel_stars = part['star'].prop("host.velocity.principal")
    vels = part['star'].prop('host.velocity.principal.cylindrical')
    return rxyz, Rxy, Phixy, Zxy, Vrxy, Vphixy, Vzxy, x, y, z, vx, vy, vz, ax, ay, az, age, mass, feh, mgfe, pos_stars, pos, dist_stars, vel_stars, vels
