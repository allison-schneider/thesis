# Look at metadata in netcdf files.

from scipy.io import netcdf

file = netcdf.netcdf_file("data/hgt-000.nc", mmap=False)
vars = file.variables
dimensions = file.dimensions
geopotential_height = vars["gh"]
elevation = vars["lev"]
test = elevation

print("The pressure level is ", elevation[0], elevation.units)