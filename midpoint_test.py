import numpy as np
import traj

initial_lat = 21
initial_lon = 100

final_lat = 60
final_lon = 320

mid_lat, mid_lon = traj.midpoint(initial_lat, initial_lon, final_lat, final_lon)

lat = np.array((initial_lat, mid_lat, final_lat))
lon = np.array((initial_lon, mid_lon, final_lon))

print(lat, lon)
traj.plot_ortho(lat, lon)