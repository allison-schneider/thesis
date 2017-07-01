# From kgoebber on github
# Calculate the distance in meters between points given by latitude and longitude.

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

lat = np.arange(90,-0.1,-0.5)
lon = np.arange(0,360.1,0.5)
print(lat)
print(lon)

dx, dy = calc_dx_dy(lon,lat)

print(dx)
print(dy)

# THE DOCSTRING LIES THIS ACCEPTS 2D ARRAYS
# def calc_dx_dy(longitude,latitude):
#     ''' This definition calculates the distance between grid points that are in
#         a latitude/longitude format.
        
#         Equations from:
#         http://andrew.hedges.name/experiments/haversine/
        
#         Accepts, 1D arrays for latitude and longitude
        
#         Returns: dx, dy; 2D arrays of distances between grid points in the x and y direction in meters 
#     '''
#     dx = np.empty(latitude.shape)
#     dy = np.zeros(longitude.shape)
    
#     dlat_x = (latitude[1:,:]-latitude[:-1,:])*np.pi/180
#     dlon_x = (longitude[1:,:]-longitude[:-1,:])*np.pi/180
    
#     dlat_y = (latitude[:,1:]-latitude[:,:-1])*np.pi/180
#     dlon_y = (longitude[:,1:]-longitude[:,:-1])*np.pi/180
    
#     #print(dlat_y.shape)
#     #print(dlon_y.shape)
#     for i in range(latitude.shape[1]):
#         for j in range(latitude.shape[0]-1):
#             a_x = (np.sin(dlat_x[j,i]/2))**2 + np.cos(latitude[j,i]*np.pi/180) * np.cos(latitude[j+1,i]*np.pi/180) * (np.sin(dlon_x[j,i]/2))**2
#             c_x = 2 * np.arctan2(np.sqrt(a_x), np.sqrt(1-a_x))
#             dx[j,i] = c_x * 6371229
#     dx[j+1,:] = dx[j,:]
    
#     for i in range(latitude.shape[1]-1):
#         for j in range(latitude.shape[0]):       
#             a_y = (np.sin(dlat_y[j,i]/2))**2 + np.cos(latitude[j,i]*np.pi/180) * np.cos(latitude[j,i+1]*np.pi/180) * (np.sin(dlon_y[j,i]/2))**2
#             c_y = 2 * np.arctan2(np.sqrt(a_y), np.sqrt(1-a_y))
#             dy[j,i] = c_y * 6371229
#     print(j,i)
#     dy[:,i+1] = dy[:,i]
    
#     return dx, dy