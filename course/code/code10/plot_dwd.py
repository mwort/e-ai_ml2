from eccodes import codes_grib_new_from_file, codes_get, codes_get_array, codes_release
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def read_grib_data(filename, shortname):
    with open(filename, "rb") as f:
        while True:
            gid = codes_grib_new_from_file(f)
            if gid is None:
                break
            try:
                if codes_get(gid, "shortName") == shortname:
                    data = codes_get_array(gid, "values")
                    codes_release(gid)
                    return data
            except Exception as e:
                print("Error reading GRIB message:", e)
                codes_release(gid)
                break
    return None

# Reading data from GRIB2 files
latitudes = read_grib_data("clat.grib2", "tlat")
longitudes = read_grib_data("clon.grib2", "tlon")
temperatures_k = read_grib_data("data.grib2", "2t")

if latitudes is None or longitudes is None or temperatures_k is None:
    print("Error retrieving data from GRIB files.")
    exit(1)

# Convert temperatures from Kelvin to Celsius
temperatures_c = temperatures_k - 273.15

# Mask out temperatures outside the range [-10, 50] Celsius
mask = (temperatures_c >= -10) & (temperatures_c <= 50)
latitudes = latitudes[mask]
longitudes = longitudes[mask]
temperatures_c = temperatures_c[mask]

# Create scatter plot
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
sc = ax.scatter(longitudes, latitudes, c=temperatures_c, cmap='jet', s=1, transform=ccrs.PlateCarree())
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.RIVERS)
ax.add_feature(cfeature.LAND, edgecolor='black')

# Add colorbar
cbar = plt.colorbar(sc, orientation='vertical', pad=0.05)
cbar.set_label('2m Temperature (°C)')

# Save plot
plt.savefig("t2m.png", dpi=300)
plt.close(fig)