# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.io import netcdf
# from mpl_toolkits.basemap import Basemap
import sys
import traj

# Command line arguments:
# 1 -- latitude in degrees, -90 to 90
# 2 -- longitude in degrees, -180 to 180
# 3 -- keyword for calculation scheme, from the following list 
#		grid -- calculate based on gridded wind fields
#		force -- calculate from geostrophic wind equation using geopotential height field

lat, lon = float(sys.argv[1]), float(sys.argv[2])
scheme = sys.argv[3]

trajectory_lat, trajectory_lon = traj.runge_kutta_trajectory(lat, lon, scheme)

# Plot trajectory
traj.plot_ortho(trajectory_lat, trajectory_lon, savefig=False)