import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf
from mpl_toolkits.basemap import Basemap
import sys
import traj

# First command line argument is latitude, second is longitude
lat, lon = float(sys.argv[1]), float(sys.argv[2])
trajectory_lat, trajectory_lon = traj.trajectory(lat, lon)
# Adjust longitude to -180 to 180 range
trajectory_lon = trajectory_lon - 180
traj.plot_ortho(trajectory_lat, trajectory_lon, savefig=False)