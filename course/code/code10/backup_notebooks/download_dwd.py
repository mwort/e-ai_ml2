import requests
import bz2
import tempfile
from eccodes import GribFile, codes_index_get

# Download the .bz2 file
url = "https://opendata.dwd.de/weather/nwp/icon-d2/grib/00/t_2m/icon-d2_germany_icosahedral_single-level_2025123100_000_2d_t_2m.grib2.bz2"
response = requests.get(url)

# Save and decompress the .bz2 file
with tempfile.NamedTemporaryFile(delete=False) as temp_file:
    temp_file.write(response.content)
    temp_file.seek(0)
    decompressed_data = bz2.decompress(temp_file.read())

# Write decompressed data to a temporary file
with tempfile.NamedTemporaryFile(delete=False) as decompressed_file:
    decompressed_file.write(decompressed_data)
    decompressed_file_path = decompressed_file.name

# Use eccodes to read the GRIB file and list all keys in the first message
with GribFile(decompressed_file_path) as f:
    message = f.next()
    keys = codes_index_get(message, 'parameterName')
    for key in keys:
        print(key)