import os
import bz2
import requests
from eccodes import codes_grib_new_from_file, codes_get, codes_release

# Download the file
url = "https://opendata.dwd.de/weather/nwp/icon-d2/grib/00/t_2m/icon-d2_germany_icosahedral_single-level_2025123100_000_2d_t_2m.grib2.bz2"
compressed_file_path = "data.grib2.bz2"
decompressed_file_path = "data.grib2"

response = requests.get(url)
with open(compressed_file_path, 'wb') as f:
    f.write(response.content)

# Decompress the .bz2 file
with bz2.BZ2File(compressed_file_path, 'rb') as file:
    with open(decompressed_file_path, 'wb') as new_file:
        new_file.write(file.read())

# Remove the compressed file
os.remove(compressed_file_path)

# Print basic info about the GRIB file
with open(decompressed_file_path, 'rb') as f:
    gid = codes_grib_new_from_file(f)
    if gid is not None:
        keys = [
            "shortName", "level", "dataDate", "dataTime", "forecastTime",
            "units", "cfVarName", "name", "typeOfLevel"
        ]
        for key in keys:
            try:
                value = codes_get(gid, key)
                print(f"{key}: {value}")
            except Exception as e:
                print(f"Could not get key {key}: {e}")
        codes_release(gid)