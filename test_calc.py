# From kgoebber on github
# Calculate the distance in meters between points given by latitude and longitude.
# I fixed it 6/30

import numpy as np

def calc_dx_dy(longitude,latitude):
    ''' This definition calculates the distance between grid points that are in
        a latitude/longitude format.
        
        Equations from:
        http://andrew.hedges.name/experiments/haversine/

        dy should be close to 55600 m
        dx at pole should be 0 m
        dx at equator should be close to 55600 m
        
        Accepts, 1D arrays for latitude and longitude
        
        Returns: dx, dy; 2D arrays of distances between grid points 
                                    in the x and y direction in meters 
    '''
    dlat = np.abs(latitude[1]-latitude[0])*np.pi/180
    dy = 2*(np.arctan2(np.sqrt((np.sin(dlat/2))**2),np.sqrt(1-(np.sin(dlat/2))**2)))*6371000
    dy = np.ones((latitude.shape[0],longitude.shape[0]))*dy

    dx = np.empty((latitude.shape))
    dlon = np.abs(longitude[1] - longitude[0])*np.pi/180
    for i in range(latitude.shape[0]):
        a = (np.cos(latitude[i]*np.pi/180)*np.cos(latitude[i]*np.pi/180)*np.sin(dlon/2))**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a) )
        dx[i] = c * 6371000
    dx = np.repeat(dx[:,np.newaxis],longitude.shape,axis=1)
    return dx, dy

lats = np.arange(90,-0.1,-0.5)
lons = np.arange(0,360.1,0.5)
print(lats)
print(lons)

dx, dy = calc_dx_dy(lons,lats)

print(dx)
print(dy)
