import urllib.request
import bz2

# Define the URLs
clat_url = "https://opendata.dwd.de/weather/nwp/icon-d2/grib/00/clat/icon-d2_germany_icosahedral_time-invariant_2025123100_000_0_clat.grib2.bz2"
clon_url = "https://opendata.dwd.de/weather/nwp/icon-d2/grib/00/clon/icon-d2_germany_icosahedral_time-invariant_2025123100_000_0_clon.grib2.bz2"

# Download the files
clat_bz2 = "clat.grib2.bz2"
clon_bz2 = "clon.grib2.bz2"
urllib.request.urlretrieve(clat_url, clat_bz2)
urllib.request.urlretrieve(clon_url, clon_bz2)

# Decompress the .bz2 files
with bz2.open(clat_bz2, 'rb') as f_in:
    with open("clat.grib2", 'wb') as f_out:
        f_out.write(f_in.read())

with bz2.open(clon_bz2, 'rb') as f_in:
    with open("clon.grib2", 'wb') as f_out:
        f_out.write(f_in.read())