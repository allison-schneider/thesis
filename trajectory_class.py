import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf
from mpl_toolkits.basemap import Basemap
import sys

class Parcel:
	def __init__(self, latitude, longitude):
		self.lat = latitude
		self.lon = longitude

p = Parcel([41, 42, 43], [-71, -72, -73])
print(p.lat)