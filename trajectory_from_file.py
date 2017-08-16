import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Global variables
EARTH_RADIUS = 6371e3    # meters

def haversine(latitude1, longitude1, latitude2, longitude2):
    """ Great-circle distance between two points. Latitudes and longitudes
    are 1D NumPy arrays. 
    Returns a 1D array of distances between two trajectories. """

    lat1 = np.radians(latitude1)
    lat2 = np.radians(latitude2)
    lon1 = np.radians(longitude1)
    lon2 = np.radians(longitude2)

    dlat = np.absolute(lat2 - lat1)
    dlon = np.absolute(lon2 - lon1)

    a = (np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) 
        * np.sin(dlon / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) 
    d = EARTH_RADIUS * c    # distance in meters
    return d

class Trajectory:
    """ Version of trajectory class for analyzing trajectory data from 
    text files."""
    
    def __init__(self,
                 scheme="grid",     # "grid" or "force"
                 timestep=90, 
                 source="model",    # "model" or "hysplit"
                 location="boston", # "boston"
                 vertical="3D"):    # "3D" or "isobaric"             
        self.scheme = scheme        # Trajectory calculation scheme
        self.timestep = timestep    # Timestep in seconds
        self.source = source        # Study models or HYSPLIT trajectories
        self.location = location    # Parcel launch site 
        self.vertical = vertical    # Vertical transport scheme

        if source == "model":
            lat_title = ("trajectory_data/{0}_latitudes_{1}_{2}.txt").format(
                self.location, self.scheme, self.timestep)
            lon_title = ("trajectory_data/{0}_longitudes_{1}_{2}.txt").format(
                self.location, self.scheme, self.timestep)
            u_title = ("trajectory_data/{0}_trajectory_u_{1}_{2}.txt").format(
                self.location, self.scheme, self.timestep)
            v_title = ("trajectory_data/{0}_trajectory_v_{1}_{2}.txt").format(
                self.location, self.scheme, self.timestep)

            self.latitudes = np.loadtxt(lat_title)
            self.longitudes = np.loadtxt(lon_title)
            self.trajectory_u = np.loadtxt(u_title)
            self.trajectory_v = np.loadtxt(v_title)

            # List of times in hours
            self.times = np.arange(np.size(self.latitudes[:,0])) * (
                self.timestep / 60 ** 2)

        if source == "hysplit":

            self.latitudes, self.longitudes, self.times = self.load_hysplit()
            self.trajectory_u = np.zeros_like(self.latitudes)
            self.trajectory_v = np.zeros_like(self.longitudes)

        self.mean_latitudes, self.mean_longitudes = self.mean_trajectory()

    def load_hysplit(self):
        # Get 1D lat and lon vectors from file
        num_trajectories = 25
        file = np.loadtxt("trajectory_data/hysplit_{0}_{1}.txt".format(
            self.vertical, self.location))
        file_time = file[:,8]    # time in hours
        file_lat = file[:,9]
        file_lon = file[:,10]

        # Initialize latitude and longitude arrays
        num_rows = np.size(file_lat) // num_trajectories
        latitude = np.zeros((num_rows, num_trajectories))
        longitude = np.zeros_like(latitude)
        time = np.zeros_like(latitude)
        
        # Separate lats and lons by trajectory
        for i in np.arange(np.size(latitude)):
            row = i // num_trajectories
            column = i % num_trajectories

            time[row, column] = file_time[i]
            latitude[row, column] = file_lat[i]
            longitude[row, column] = file_lon[i]

        return latitude, longitude, time

    def mean_trajectory(self):
        """ Get the centroid of parcels at each timestep. """

        # Convert latitudes and longitudes to Cartesian coordinates
        x = (np.cos(np.radians(self.latitudes)) * 
            np.cos(np.radians(self.longitudes)))
        y = (np.cos(np.radians(self.latitudes)) * 
            np.sin(np.radians(self.longitudes)))
        z = np.sin(np.radians(self.latitudes))

        # Get average x, y, z values
        mean_x = np.mean(x, axis=1)
        mean_y = np.mean(y, axis=1)
        mean_z = np.mean(z, axis=1)

        # Convert average values to trajectory latitudes and longitudes
        mean_longitudes = np.degrees(np.arctan2(mean_y, mean_x))
        hypotenuse = np.sqrt(mean_x ** 2 + mean_y ** 2)
        mean_latitudes = np.degrees(np.arctan2(mean_z, hypotenuse))

        return mean_latitudes, mean_longitudes

    def rms_distance(self):
        """ Calculate the root mean square distance of each trajectory from the
        mean trajectory. """

        # Make mean lat and lon arrays the same shape as trajectory arrays
        mean_lat = np.repeat(self.mean_latitudes[:, np.newaxis], 
                             np.size(self.latitudes, axis=1), axis=1)
        mean_lon = np.repeat(self.mean_longitudes[:, np.newaxis], 
                             np.size(self.longitudes, axis=1), axis=1)

        rms = np.sqrt(np.mean(haversine(mean_lat, mean_lon, 
                              self.latitudes, self.longitudes) ** 2, axis=1))

        return rms

    def plot_ortho(self, lat_center=90, lon_center=-50, savefig=False):
        """ Orthographic projection plot."""
        map = Basemap(projection='ortho', lon_0=lon_center, lat_0=lat_center, 
                        resolution='c')
        map.drawcoastlines(linewidth=0.25, color='gray')
        map.drawcountries(linewidth=0)
        map.fillcontinents(color='white',lake_color='white', zorder=1)
        # draw the edge of the map projection region (the projection limb)
        map.drawmapboundary(fill_color='white')
        map.plot(self.longitudes, self.latitudes,
                 latlon=True, zorder=2, color='black')
        #map.plot(self.mean_longitudes, self.mean_latitudes,
        #         latlon=True, zorder=2, color='green')
        if savefig == True:
            filename1 = "plots/friction_180.pdf"
            plt.savefig(filename1)

        plt.show()
        return map

    def graph_speed(self):
        """ Graph of u and v along the trajectory. """
        lat_length = 111.32e3       # Length of a degree of latitude in meters
        time = np.arange(np.size(self.latitudes[:,0])) * (self.timestep 
                                                           / (60 ** 2 * 24))
        v_threshold = ((0.75 * 0.25 * lat_length) / self.timestep 
                        * np.ones(np.size(time)))
        v_threshold = v_threshold[:, np.newaxis]
        v_diff = v_threshold - self.trajectory_v

        u_threshold = (0.75 * 0.25 * lat_length 
            * np.cos(np.radians(self.latitudes))) / self.timestep
        u_diff = u_threshold - self.trajectory_u

        zero = np.zeros(np.size(time))
        
        # new style method 1; unpack the axes
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
        u_line = ax1.plot(time, u_diff, color="black", linewidth=1)
        u_zero_line = ax1.plot(time, zero, color="black", linestyle="--",
                                linewidth=2)
        ax1.set_title("Dynamic Trajectory Zonal Speeds")

        v_line = ax2.plot(time, v_diff, color="black", linewidth=1)
        v_zero_line = ax2.plot(time, zero, color="black", linestyle="--",
                                linewidth=2)
        ax2.set_title("Dynamic Trajectory Meridional Speeds")

        plt.xlabel("Time in days")
        plt.ylabel("                                         " #label padding
            "Velocity in m/s")
        #plt.savefig("plots/timestep_friction_120.eps")
        #plt.show()
        return ax1, ax2

    def threshold(self):
        """ Difference between model and HYSPLIT threshold speeds. """
        lat_length = 111.32e3       # Length of a degree of latitude in meters
        time = np.arange(np.size(self.latitudes[:,0])) * (self.timestep 
                                                           / (60 ** 2 * 24))
        v_threshold = ((0.75 * 0.25 * lat_length) / self.timestep 
                        * np.ones(np.size(time)))
        v_threshold = v_threshold[:, np.newaxis]
        v_diff = v_threshold - self.trajectory_v

        u_threshold = (0.75 * 0.25 * lat_length 
            * np.cos(np.radians(self.latitudes))) / self.timestep
        u_diff = u_threshold - self.trajectory_u
        return u_diff, v_diff, time

    def graph_rms(self):
        """ Graph of rms distance from mean trajectory. """
        rms = self.rms_distance()

        fig = plt.figure()
        ax2 = fig.add_subplot(1, 1, 1)

        rms_line, = ax2.plot(self.times, rms)
        #t_half_line, = ax2.plot(self.times, self.times ** 0.5)
        
        ax2.set_title("RMS Distance from Mean Trajectory")
        ax2.set_xlabel("Time (hours)")
        ax2.set_ylabel("RMS distance (m)")
        plt.show()
        return rms[-1]

def speed_subplots():
    """ Graph u and v speeds for two schemes. """
    trajectory_friction = Trajectory(scheme="friction", timestep=90, 
        source="model")
    trajectory_grid = Trajectory(scheme="grid", timestep=90, source="model")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True, 
        figsize=(10, 6))
    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', 
        right='off')

    (u_diff_grid, v_diff_grid, time) = trajectory_grid.threshold()
    (u_diff_friction, v_diff_friction, time) = trajectory_friction.threshold()
    zero = np.zeros(np.size(u_diff_grid[:,0]))

    ax1.plot(time, -u_diff_grid, color="black", linewidth=1)
    ax1.plot(time, zero, color="black", linestyle="--", linewidth=2)
    ax1.set_title("Kinematic Trajectory Zonal Speeds", fontsize=12)
    ax1.set_ylim(-290, 45)

    ax2.plot(time, -u_diff_friction, color="black", linewidth=1)
    ax2.plot(time, zero, color="black", linestyle="--", linewidth=2)
    ax2.set_title("Dynamic Trajectory Zonal Speeds", fontsize=12)
    ax2.set_ylim(-290, 45)

    ax3.plot(time, -v_diff_grid, color="black", linewidth=1)
    ax3.plot(time, zero, color="black", linestyle="--", linewidth=2)
    ax3.set_title("Kinematic Trajectory Meridional Speeds", fontsize=12)
    ax3.set_ylim(-290, 45)

    ax4.plot(time, -v_diff_friction, color="black", linewidth=1)
    ax4.plot(time, zero, color="black", linestyle="--", linewidth=2)
    ax4.set_title("Dynamic Trajectory Meridional Speeds", fontsize=12)
    ax4.set_ylim(-290, 45)

    plt.xlabel("Time (days)")
    plt.ylabel("Velocity (m/s)              ")

    plt.savefig("plots/speed_subplots_180_negative.pdf")

    plt.show()

def deviation():
    scheme = "friction"
    location = "barau"
    vertical = "isobaric"
    num_trajectories = 25

    experimental = Trajectory(scheme=scheme, timestep=90, source="model", 
        location=location)
    reference = Trajectory(scheme=scheme, timestep=90, source="hysplit", 
        location=location, vertical=vertical)

    # Find indices of times in experimental that match times in reference
    time_index = np.searchsorted(experimental.times, reference.times)[:,0]

    # Select times, latitudes and longitudes from experimental in reference
    reduced_times = experimental.times[time_index]
    reduced_latitudes = experimental.latitudes[time_index]
    reduced_longitudes = experimental.longitudes[time_index]

    distance = haversine(reduced_latitudes, reduced_longitudes,
        reference.latitudes, reference.longitudes)

    ahtd = np.sqrt(np.sum(distance ** 2, axis=1)) / num_trajectories

    reference_distance = haversine(reference.latitudes[1:,:], 
        reference.longitudes[1:,:], reference.latitudes[:-1,:],
        reference.longitudes[:-1,:]) 

    l_h = (np.sum(np.sqrt(np.sum(reference_distance ** 2, axis=1))) 
        / num_trajectories)

    rhtd = ahtd / l_h

    return rhtd

def reference_deviation():
    location = "boston"
    num_trajectories = 25

    reference_3d = Trajectory(timestep=90, source="hysplit", 
        location=location, vertical="3D")
    reference_iso = Trajectory(timestep=90, source="hysplit", 
        location=location, vertical="isobaric")

    distance = haversine(reference_3d.latitudes, reference_3d.longitudes,
        reference_iso.latitudes, reference_iso.longitudes)

    ahtd = np.sqrt(np.sum(distance ** 2, axis=1)) / num_trajectories

    reference_distance = haversine(reference_3d.latitudes[1:,:], 
        reference_3d.longitudes[1:,:], reference_iso.latitudes[:-1,:],
        reference_iso.longitudes[:-1,:]) 

    l_h = (np.sum(np.sqrt(np.sum(reference_distance ** 2, axis=1))) 
        / num_trajectories)

    rhtd = ahtd / l_h

    return rhtd

rhtd = deviation()
print("rhtd is \n", rhtd)
print("last rhtd is", rhtd[-1])

#tra = Trajectory(scheme="friction", timestep=90, source="model", 
#    location="boston", vertical="3D")
#tra.plot_ortho()
#last_rms = tra.graph_rms()
# print("Last RMSE value is", last_rms)