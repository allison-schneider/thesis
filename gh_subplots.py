# Only works in Python 2

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf
from mpl_toolkits.basemap import Basemap

num_rows = 4
num_columns = 2

fig, axes = plt.subplots(4, 2, figsize=(10, 10))
# add a big axis, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', 
    right='off')

for i in np.arange(8):
    row = i % num_rows
    column = i // num_rows

    hour = i * 24
    filename = "data/hgt-{:03d}.nc".format(hour)
    file = netcdf.netcdf_file(filename, mmap=False)
    vars = file.variables
    file.close()

    # Get geopotential height
    lat = vars["lat"][:][:]
    lon = vars["lon"][:][:] % 360
    gh = vars["gh"][0][0]

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    m = Basemap(projection="cyl", llcrnrlat=-90, urcrnrlat=90,
            llcrnrlon=-180, urcrnrlon=180, resolution='c', ax=axes[row, column])
    m.drawcoastlines(linewidth=0.3, zorder=3)
    m.fillcontinents(color="white", lake_color="white", zorder=0)
    m.drawmapboundary(fill_color="white", zorder=0)
    # draw parallels
    parallels = np.arange(-90, 90, 30.)
    m.drawparallels(parallels, labels=[1,0,0,0], linewidth=0.25, fontsize=8)
    # draw meridians
    meridians = np.arange(0, 360., 30.)
    m.drawmeridians(meridians, labels=[0,0,0,0], linewidth=0.25, fontsize=8)
    gh = m.pcolormesh(lon_grid, lat_grid, gh, latlon=True, 
        zorder=2, cmap="plasma", vmin=9300, vmax=11000)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(gh, cax=cbar_ax)

plt.show()