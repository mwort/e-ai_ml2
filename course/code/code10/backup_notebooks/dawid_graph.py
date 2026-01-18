import os
from dotenv import load_dotenv
import sys

import os

# DWD proxy
os.environ["HTTP_PROXY"]  = "http://ofsquid.dwd.de:8080"
os.environ["HTTPS_PROXY"] = "http://ofsquid.dwd.de:8080"

# Optional but recommended
os.environ["http_proxy"]  = os.environ["HTTP_PROXY"]
os.environ["https_proxy"] = os.environ["HTTPS_PROXY"]

# Load from .env file
load_dotenv()

# Retrieve the key
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    print("✅ OpenAI key loaded.")
    os.environ["OPENAI_API_KEY"] = openai_key
else:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file.")

# Initialize LLM
from langchain_openai import ChatOpenAI  # Updated import
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

from langgraph.graph import StateGraph
from typing import Any, TypedDict
import datetime
import pytz
import requests
from langchain_openai import ChatOpenAI

#======================================================================
# Langgraph State Class
#======================================================================
# Define LangGraph schema
class MyState(TypedDict):
    query: str
    task: str
    output: str
    # 
    # FORECASTING
    fc_datetime: str
    fc_reference_datetime: str
    fc_leadtime: str
    fc_plot_option: str
    fc_location_of_interest: str
    fc_variable: str
    fc_lat: str
    fc_lon: str
    temperature_data: Any
    #
    # CODING
    code_task: str
    code_full: str
    code_output: str
    #
    # INFORMATION
    info_topic: str

#======================================================================
# Functions
#======================================================================
from IPython.display import display, Markdown

def debug_message(text):
    display(Markdown(f"<div style='color:gray; font-style:italic; margin-left:2cm'>{text}</div>"))

#----------------------------------------------------------------------------------
#
#----------------------------------------------------------------------------------
# Function to extract forecast datetime from user input with the current time
def extract_forecast_datetime(user_input: str) -> str:
    """
    Use OpenAI to extract the forecast datetime from the user input, 
    with the current datetime provided as context for LLM.
    """
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H")
    
    # Prompt to include the current time
    prompt = f"""
    The current time is: {current_time}.
    Extract the date and time mentioned in the following user input that represents 
    the forecast datetime, give nothing else back, only the date-time in format %Y-%m-%d %H
    User input: {user_input}
    Forecast datetime:
    """
    
    response = llm.invoke(prompt)
    debug_message(f"extract_forecast_datetime: {response.content}")
    return response.content.strip()  # Clean up the response

#----------------------------------------------------------------------------------
#
#----------------------------------------------------------------------------------
from datetime import datetime, timezone

# Function to calculate forecast lead time
def calculate_forecast_lead_time(forecast_datetime: str, fc_reference_datetime: str) -> int:
    """
    Function to calculate the forecast lead time, i.e. the number of hours between
    the start of the forecast and the targeted time. 
    
    :param forecast_datetime: The datetime of the forecast, in "YYYY-MM-DD HH" format.
    :param fc_reference_datetime: The reference time when the forecast was made, in "YYYY-MM-DD HH" format.
    :return: The forecast lead time in hours (integer).
    """
    # Current time (timezone-aware)
    current_time = datetime.now(timezone.utc)
    
    # Parse the forecast and reference datetime
    forecast_time = datetime.strptime(forecast_datetime, "%Y-%m-%d %H")
    forecast_time = forecast_time.replace(tzinfo=timezone.utc)
    
    reference_time = datetime.strptime(fc_reference_datetime, "%Y-%m-%d %H")
    reference_time = reference_time.replace(tzinfo=timezone.utc)
    
    # Calculate the lead time in hours
    lead_time = (forecast_time - reference_time).total_seconds() / 3600
    
    return int(lead_time)

#----------------------------------------------------------------------------------
#
#----------------------------------------------------------------------------------
import requests
import bz2
import re
from bs4 import BeautifulSoup
import xarray as xr
import datetime

def download_icon_2m_temperature(fc_leadtime: str = "000") -> xr.DataArray:
    """
    Downloads the latest ICON-EU 2m temperature GRIB2 file (forecast hour '000' by default),
    decompresses it, and returns the temperature field as xarray.DataArray (in °C).
    """
    
    # If the forecast leadtime is greater than 78, adjust to the nearest multiple of 3
    if int(fc_leadtime) > 78:
        fc_leadtime2 = str(int(fc_leadtime) // 3 * 3)  # Nearest multiple of 3
    else:
        fc_leadtime2 = fc_leadtime

    base_url = "https://opendata.dwd.de/weather/nwp/icon-eu/grib/00/t_2m/"
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, "lxml")

    # Ensure lead time is zero-padded
    fc_leadtime2 = f"{int(fc_leadtime2):03d}"

    pattern = re.compile(rf"icon-eu_europe_regular-lat-lon_single-level_(\d{{10}})_{fc_leadtime2}_T_2M\.grib2\.bz2")
    latest_file = None
    latest_timestamp = None

    for link in soup.find_all("a"):
        href = link.get("href")
        match = pattern.match(href)
        if match:
            rundate = match.group(1)
            latest_file = href
            latest_timestamp = rundate
            break

    if latest_file is None:
        raise Exception(f"No file with forecast hour {fc_leadtime2} found.")

    # Download & decompress
    url = base_url + latest_file
    debug_message(f"Downloading: {url}")
    r = requests.get(url)
    local_bz2 = "downloads/temp.grib2.bz2"
    local_grib = "downloads/temp.grib2"
    with open(local_bz2, "wb") as f:
        f.write(r.content)

    with bz2.open(local_bz2, 'rb') as f_in, open(local_grib, 'wb') as f_out:
        f_out.write(f_in.read())

    # Read using cfgrib
    ds = xr.open_dataset(local_grib, engine="cfgrib", backend_kwargs={
        "indexpath": "",
        "decode_timedelta": True
    })    
    t2m = ds["t2m"] - 273.15  # Kelvin → °C

    t2m.name = "2m_temperature"
    t2m.attrs["units"] = "degC"
    t2m.attrs["fc_reference_datetime"] = datetime.datetime.strptime(latest_timestamp, "%Y%m%d%H")
    t2m.attrs["fc_leadtime"] = fc_leadtime

    return t2m

#----------------------------------------------------------------------------------
#
#----------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import LinearSegmentedColormap
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import datetime

def plot_t2m_EU(t2m, projection_name="PlateCarree", save_plot=False):
    """
    Plots 2m temperature field over Europe using a selected Cartopy projection.
    Available options: "PlateCarree", "LambertConformal", "AlbersEqualArea", "Mercator"
    
    Arguments:
    - t2m: Temperature data to be plotted (xarray.DataArray)
    - projection_name: The projection to use for the map (default is 'PlateCarree')
    - save_plot: If True, saves the plot as an image file (default is False)
    """
    
    # Projections
    projections = {
        "PlateCarree": ccrs.PlateCarree(),
        "LambertConformal": ccrs.LambertConformal(central_longitude=10, central_latitude=50),
        "AlbersEqualArea": ccrs.AlbersEqualArea(central_longitude=10, central_latitude=50),
        "Mercator": ccrs.Mercator()
    }

    if projection_name not in projections:
        raise ValueError(f"Invalid projection '{projection_name}'. Choose from: {list(projections.keys())}")

    proj = projections[projection_name]

    # Colormap definition
    temp_points = [-20, 0, 10, 18, 25, 30, 35, 40]
    colors = [
        '#081d58',  # very cold (deep blue)
        '#00BFFF',  # clear blue
        '#48D1CC',  # teal (greenish blue)
        '#00ff66',  # bright green
        '#ffff33',  # strong yellow
        '#ff9900',  # orange
        '#ff0000',  # red
        '#7f0000',  # dark red
    ]
    vmin, vmax = -20, 40
    normed_points = [(t - vmin) / (vmax - vmin) for t in temp_points]
    cmap = LinearSegmentedColormap.from_list("temp_smooth_custom", list(zip(normed_points, colors)))

    # Plot setup
    plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=proj)
    ax.set_extent([-25, 45, 30, 72], crs=ccrs.PlateCarree())

    # Plot the temperature field
    t2m.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kwargs={"label": "2m Temperature (°C)", "shrink": 0.95}
    )

    # Coastlines and gridlines
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True, linestyle=':', color='gray', linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # Title from metadata
    title = "ICON-EU 2m Temperature (°C)"
    attrs = t2m.attrs
    if "fc_reference_datetime" in attrs and "fc_leadtime" in attrs:
        ref_time = attrs["fc_reference_datetime"]
        fc_leadtime = int(attrs["fc_leadtime"])
        valid_time = ref_time + datetime.timedelta(hours=fc_leadtime)
        title += f"\n{ref_time.strftime('%Y-%m-%d %H:%M')} UTC +{fc_leadtime:03d}h; forecast for: {valid_time.strftime('%Y-%m-%d %H:%M')} UTC"

    plt.title(title)

    # Show or save the plot
    if save_plot:
        plot_filename = "temperature_forecast.png"
        plt.savefig(plot_filename, dpi=150)
        debug_message(f"Plot saved as {plot_filename}")
    else:
        plt.show()

    return f"Plot created and displayed (saved: {save_plot})"

#----------------------------------------------------------------------------------
#
#----------------------------------------------------------------------------------
from scipy.interpolate import RegularGridInterpolator

def interpolate_t2m(t2m, lat: float, lon: float) -> float:
    """
    Interpolates the 2m temperature field at the given latitude and longitude.
    Returns the interpolated temperature in °C.
    """
    # Ensure latitude is increasing (needed by interpolator)
    if t2m.latitude.values[0] > t2m.latitude.values[-1]:
        t2m = t2m.sortby("latitude")
    
    # Ensure longitude is increasing
    if t2m.longitude.values[0] > t2m.longitude.values[-1]:
        t2m = t2m.sortby("longitude")

    # Build interpolator
    interpolator = RegularGridInterpolator(
        (t2m.latitude.values, t2m.longitude.values),
        t2m.values,
        bounds_error=False,
        fill_value=np.nan
    )

    # Interpolate at requested location
    interpolated_value = interpolator((lat, lon))
    return float(interpolated_value)

#----------------------------------------------------------------------------------
#
#----------------------------------------------------------------------------------
from geopy.geocoders import Nominatim

def get_coordinates_from_name(place_name: str) -> tuple[float, float]:
    """
    Returns (latitude, longitude) of a place using OpenStreetMap's Nominatim service.
    """
    geolocator = Nominatim(user_agent="icon-eu-weather-tool")
    location = geolocator.geocode(place_name)

    if location is None:
        raise ValueError(f"Place '{place_name}' not found.")

    return (location.latitude, location.longitude)

#----------------------------------------------------------------------------------
#
#----------------------------------------------------------------------------------
import requests
from bs4 import BeautifulSoup
import re

def get_latest_forecast_reference_time() -> str:
    """
    Scrapes the DWD open data server for the latest forecast reference time.
    
    :return: The latest forecast reference time in "YYYY-MM-DD HH" format.
    """
    base_url = "https://opendata.dwd.de/weather/nwp/icon-eu/grib/00/t_2m/"
    response = requests.get(base_url)
    
    if response.status_code != 200:
        raise Exception("Failed to retrieve data from DWD server.")
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Regular expression to extract forecast reference time from filenames
    pattern = re.compile(r"icon-eu_europe_regular-lat-lon_single-level_(\d{10})_\d{3}_T_2M\.grib2\.bz2")
    
    latest_file = None
    latest_timestamp = None

    for link in soup.find_all("a"):
        href = link.get("href")
        match = pattern.match(href)
        if match:
            timestamp = match.group(1)  # Extract timestamp from filename
            if latest_timestamp is None or timestamp > latest_timestamp:
                latest_file = href
                latest_timestamp = timestamp
    
    if latest_file is None:
        raise Exception("No forecast data found on the server.")
    
    # Convert timestamp to desired format (YYYY-MM-DD HH)
    forecast_reference_time = f"{latest_timestamp[:4]}-{latest_timestamp[4:6]}-{latest_timestamp[6:8]} {latest_timestamp[8:10]}"
    
    debug_message(f"get_latest_forecast_reference_time: {forecast_reference_time}")
    return forecast_reference_time

#----------------------------------------------------------------------------------
#
#----------------------------------------------------------------------------------
import datetime
import pytz

def calculate_forecast_lead_time(forecast_datetime: str, reference_datetime: str) -> int:
    """
    Calculates the forecast lead time in hours from the forecast reference time to the requested forecast time.
    Forecast time must be within 120 hours of the reference time.

    :param forecast_datetime: A string representing the forecast datetime (in "YYYY-MM-DD HH" format).
    :param reference_datetime: A string representing the reference datetime when the forecast was made.
    :return: The lead time in hours (integer).
    :raises ValueError: If the forecast datetime is more than 120 hours ahead of the reference time.
    """
    # Convert the reference datetime into a datetime object (timezone-aware)
    reference_time = datetime.datetime.strptime(reference_datetime, "%Y-%m-%d %H")
    reference_time = pytz.utc.localize(reference_time)  # Localize to UTC

    # Convert the forecast string into a datetime object (timezone-aware)
    forecast_time = datetime.datetime.strptime(forecast_datetime, "%Y-%m-%d %H")
    forecast_time = pytz.utc.localize(forecast_time)  # Localize to UTC
    
    # Calculate the difference in hours
    lead_time = (forecast_time - reference_time).total_seconds() / 3600
    
    if lead_time > 120:
        raise ValueError(f"Forecast lead time exceeds 120 hours: {lead_time:.2f} hours")

    debug_message(f"calculate_forecast_lead_time: {lead_time}")
    return int(lead_time)

#----------------------------------------------------------------------------------
#
#----------------------------------------------------------------------------------
# Function to extract the location from user input
def extract_location(user_input: str) -> str:
    """
    Extracts the location (e.g., town, city) from the user's input using OpenAI's model.

    This function communicates with the OpenAI model to identify a location mentioned in the user's input.
    The model is tasked with recognizing geographical names such as cities, towns, or specific places mentioned 
    in the query and returning it as a cleanly formatted string. The location is expected to be mentioned explicitly
    in the user input.

    Example:
    If the user input is:
        "What is the weather like in Berlin tomorrow?"
    The model should return:
        "Berlin"

    If no location is specified, the model may return an empty string or a default value indicating the absence of 
    location data.

    Parameters:
    user_input (str): The raw user input string from which the location needs to be extracted.

    Returns:
    str: The location (e.g., city, town) extracted from the input, or an empty string if no location is identified.
    """
    prompt = f"""
    Extract the location (e.g., town, city) mentioned in the following user input:
    User input: {user_input}
    Location:
    """
    response = llm.invoke(prompt)
    debug_message(f"extract_location: {response.content}")
    return response.content.strip()  # Clean up the response

#----------------------------------------------------------------------------------
#
#----------------------------------------------------------------------------------
# Function to extract the weather variable (e.g., temperature, humidity) from user input
def extract_variable(user_input: str) -> str:
    """
    Extracts the weather variable (e.g., temperature, humidity, wind speed) from the user's input using OpenAI's model.

    This function uses OpenAI's model to parse the user's query and identify any weather-related variables mentioned in it. 
    Common variables include, but are not limited to, temperature, humidity, precipitation, wind speed, etc. The model 
    will return the specific weather variable that the user is asking about. If no weather variable is explicitly mentioned, 
    the model will return an empty string or a default value.

    Example:
    If the user input is:
        "What is the humidity in Berlin tomorrow?"
    The model should return:
        "humidity"
    
    If the user input is:
        "What is the temperature like tomorrow?"
    The model should return:
        "temperature"

    Parameters:
    user_input (str): The raw user input string from which the weather variable needs to be extracted.

    Returns:
    str: The weather variable (e.g., temperature, humidity, wind speed) extracted from the input, or an empty string 
         if no weather variable is found.
    """
    prompt = f"""
    Extract the weather variable (e.g., temperature, humidity, wind speed) mentioned in the following user input:
    User input: {user_input}
    Variable:
    """
    response = llm.invoke(prompt)
    debug_message(f"extract_variable: {response.content}")
    return response.content.strip()  # Clean up the response

#----------------------------------------------------------------------------------
#
#----------------------------------------------------------------------------------
import inspect
import sys

def list_functions():
    """
    Lists all functions in the current module (dawid2) with their docstrings.
    """
    functions = []
    
    # Get the current module from sys.modules
    current_module = sys.modules[__name__]
    
    # Use inspect to get all callable members in the current module
    for name, obj in inspect.getmembers(current_module):
        if inspect.isfunction(obj):  # Check if it's a function
            docstring = inspect.getdoc(obj) or "No description available."
            functions.append((name, docstring))
    
    return functions

#----------------------------------------------------------------------------------
#
#----------------------------------------------------------------------------------


#----------------------------------------------------------------------------------
#
#----------------------------------------------------------------------------------

#======================================================================
# Langgraph State Class
#======================================================================
from langgraph.graph import StateGraph

# Function to extract forecast datetime
def extract_forecast_datetime_node(state: MyState) -> MyState:
    user_input = state["query"]
    forecast_datetime = extract_forecast_datetime(user_input)
    state["fc_datetime"] = forecast_datetime
    return state

# Function to get latest forecast reference time
def get_latest_forecast_reference_time_node(state: MyState) -> MyState:
    reference_time = get_latest_forecast_reference_time()
    state["fc_reference_datetime"] = reference_time
    return state

# Function to calculate forecast lead time
def calculate_lead_time_node(state: MyState) -> MyState:
    forecast_datetime = state["fc_datetime"]
    fc_reference_datetime = state["fc_reference_datetime"]
    lead_time = calculate_forecast_lead_time(forecast_datetime, fc_reference_datetime)
    state["fc_leadtime"] = str(lead_time).zfill(3)
    return state

# Function to extract location from user input
def extract_location_node(state: MyState) -> MyState:
    user_input = state["query"]
    location = extract_location(user_input)
    state["fc_location_of_interest"] = location
    return state

# Function to extract weather variable
def extract_variable_node(state: MyState) -> MyState:
    user_input = state["query"]
    variable = extract_variable(user_input)
    state["fc_variable"] = variable
    return state

# Function to download temperature data
def download_temperature_node(state: MyState) -> MyState:
    fc_leadtime = state.get("fc_leadtime", "000")
    temperature_data = download_icon_2m_temperature(fc_leadtime)
    state["temperature_data"] = temperature_data
    return state

# Function to plot temperature data
def plot_temperature_node(state: MyState) -> MyState:
    temperature_data = state.get("temperature_data")
    if temperature_data is not None:
        plot_result = plot_t2m_EU(temperature_data, save_plot=True)
        state["output"] = plot_result
    else:
        state["output"] = "No temperature data available."
    return state

def get_coordinates_from_name_node(state: MyState) -> MyState:
    place = state.get("fc_location_of_interest", "")
    if place and place.lower() != "location: not specified":
        lat, lon = get_coordinates_from_name(place)
        state["fc_lat"] = str(lat)
        state["fc_lon"] = str(lon)
    else:
        state["fc_lat"] = ""
        state["fc_lon"] = ""
    return state

def interpolate_temperature_node(state: MyState) -> MyState:
    t2m = state.get("temperature_data")
    if not t2m:
        state["output"] = "No temperature data to interpolate."
        return state

    lat = state.get("fc_lat")
    lon = state.get("fc_lon")

    if not lat or not lon:
        state["output"] = "Latitude and longitude not set for interpolation."
        return state

    try:
        value = interpolate_t2m(t2m, float(lat), float(lon))
        state["output"] = f"Temperature at {lat}, {lon} is {value:.2f}°C"
    except Exception as e:
        state["output"] = f"Interpolation failed: {e}"

    return state

# Define the LangGraph schema and state
builder = StateGraph(state_schema=MyState)

# Add nodes (ohne output_keys)
builder.add_node("extract_forecast_datetime", extract_forecast_datetime_node)
builder.add_node("get_latest_forecast_reference_time", get_latest_forecast_reference_time_node)
builder.add_node("calculate_lead_time", calculate_lead_time_node)
builder.add_node("extract_location", extract_location_node)
builder.add_node("extract_variable", extract_variable_node)
builder.add_node("download_temperature", download_temperature_node)
builder.add_node("plot_temperature", plot_temperature_node)
#builder.add_node("get_coordinates", get_coordinates_from_name_node)
#builder.add_node("interpolate_temperature", interpolate_temperature_node)

# Set entry and finish points
builder.set_entry_point("extract_forecast_datetime")
builder.add_edge("extract_forecast_datetime", "get_latest_forecast_reference_time")
builder.add_edge("get_latest_forecast_reference_time", "calculate_lead_time")
builder.add_edge("calculate_lead_time", "extract_location")
builder.add_edge("extract_location", "extract_variable")
builder.add_edge("extract_variable", "download_temperature")
builder.add_edge("download_temperature", "plot_temperature")
#builder.add_edge("download_temperature", "get_coordinates")
#builder.add_edge("get_coordinates", "interpolate_temperature")
builder.set_finish_point("plot_temperature")
#builder.set_finish_point("interpolate_temperature")

# Compile the graph
graph = builder.compile()

#======================================================================
# Main function 
#======================================================================
from IPython.display import Markdown, display

def ai(query):
    # Initialize state
    state = {
        "query": query,
        "task": "fc",
        "output": "",
        "fc_datetime": "",
        "fc_reference_datetime": "",
        "fc_leadtime": "",
        "fc_plot_option": "full",
        "fc_location_of_interest": "",
        "fc_variable": "",
        "temperature_data": None,
        "code_task": "",
        "code_full": "",
        "code_output": "",
        "info_topic": ""
    }

    # Run the LangGraph
    result = graph.invoke(state)

    # Generate a summary using the LLM
    summary_prompt = f"""
    You are DAWID, the friendly assistant of Deutscher Wetterdienst. The user asked: "{state["query"]}"

    Intermediate results:
    - Forecast datetime: {result['fc_datetime']}
    - Forecast reference: {result['fc_reference_datetime']}
    - Lead time: {result['fc_leadtime']} hours
    - Location: {result['fc_location_of_interest']}
    - Variable: {result['fc_variable']}

    Final output: {result['output']}

    Summarize this in one or two friendly sentences for the user.
    """

    response = llm.invoke(summary_prompt)
    display(Markdown(response.content))

