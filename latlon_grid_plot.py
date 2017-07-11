import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf
from mpl_toolkits.basemap import Basemap

file = netcdf.netcdf_file("data/hgt-000.nc", mmap=False)
vars = file.variables
file.close

lat = vars['lat'][:]
lon = vars['lon'][:]

map = Basemap(projection="ortho", lat_0=45, lon_0=-100, resolution="c")
map.drawcoastlines(linewidth=0.25, color='gray')
map.drawcountries(linewidth=0)
map.fillcontinents(color='white',lake_color='white', zorder=1)
# draw the edge of the map projection region (the projection limb)
map.drawmapboundary(fill_color='white')

map.drawmeridians(lon[::30])
map.drawparallels(lat[::30])

plt.show()