import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import glob
import regionmask
import xarray as xr
import numpy as np
import rioxarray
import imageio
from rasterio import features
from shapely.vectorized import contains
from shapely.geometry import box
from shapely.geometry import Polygon, MultiPolygon
import time
from datetime import datetime
import argparse

def load_Xyrs_winter_dataset(start_winter_year, X):
    print("Loading multi-year winter dataset...")
    load_start = time.time()
    months_first_year = ['11', '12']
    months_second_year = ['01', '02', '03']
    all_file_list = []

    for i in range(X):
        year1 = start_winter_year + i
        year2 = year1 + 1
        path1 = f'/import/beegfs/CMIP6/wrf_era5/04km/{year1}'
        path2 = f'/import/beegfs/CMIP6/wrf_era5/04km/{year2}'

        for month in months_first_year:
            pattern = f"era5_wrf_dscale_4km_{year1}-{month}-*.nc"
            file_list = sorted(glob.glob(os.path.join(path1, pattern)))
            if file_list:
                all_file_list.extend(file_list)
            else:
                print(f"{year1}-{month}: ERROR No files found")

        for month in months_second_year:
            pattern = f"era5_wrf_dscale_4km_{year2}-{month}-*.nc"
            file_list = sorted(glob.glob(os.path.join(path2, pattern)))
            if file_list:
                all_file_list.extend(file_list)
            else:
                print(f"{year2}-{month}: ERROR No files found")

    load_end = time.time()
    print("Time to load X-year dataset (min):", round((load_end - load_start) / 60, 2))

    def select_vars(ds):
        return ds[['T2', 'SNOW', 'acsnow', 'rainnc', 'temp', 'XLAT', 'XLONG','Time']]

    if all_file_list:
        data = xr.open_mfdataset(all_file_list, combine='by_coords', preprocess=select_vars)
        print("X-year winter dataset opened and combined (selected variables only)!")
        return data
    else:
        raise ValueError("No files found across the X-year winter period!")


def land_mask(data, lat, lon):
    os.chdir("/import/beegfs/CMIP6/wrf_era5")
    geo_em_path = "geo_em.d02.nc"
    geo = xr.open_dataset(geo_em_path)
    landmask = geo['LANDMASK']
    landmask = landmask.squeeze(dim="Time")
    landmask_expanded = landmask.expand_dims(Time=data.Time)

    data_fixed = data.where(landmask_expanded == 1)
    data_fixed = data_fixed.chunk(dict(south_north=-1, west_east=-1))
    data_filled = data_fixed.interpolate_na(dim="south_north", method="nearest", fill_value="extrapolate")
    data_filled = data_filled.interpolate_na(dim="west_east", method="nearest", fill_value="extrapolate")
    data_final = data_filled.where(~np.isnan(landmask_expanded))
    print("Land Mask and Alaska Mask Completed")
    
    #land = shapefile.to_crs(epsg=4326)
    #alaska_bbox = box(-179, 50, -120, 72)
    #bbox_gdf = gpd.GeoDataFrame({"geometry": [alaska_bbox]}, crs="EPSG:4326")
    #alaska_land = gpd.clip(land, bbox_gdf)
    #mask_all_AK=  regionmask.mask_geopandas(alaska_land, lon,lat)

    #mask_da = xr.DataArray(mask_all_AK, coords={"south_north": data.south_north, "west_east": data.west_east},dims=["south_north", "west_east"])
    #data_masked = data_final.where(~np.isnan(mask_da))
    oceanmask=geo['LU_INDEX']
    oceanmask=oceanmask.squeeze(dim='Time')
    oceanmask_expanded = oceanmask.expand_dims(Time=data.Time)
    
    data_masked = data_final.where(oceanmask_expanded != 17)
    return data_masked


def get_winter_season_labels(time_index):
    labels = []
    for t in pd.to_datetime(time_index):
        if t.month in [11, 12]:
            labels.append(f"{t.year}-{t.year + 1}")
        elif t.month in [1, 2, 3]:
            labels.append(f"{t.year - 1}-{t.year}")
        else:
            labels.append(None)
    return np.array(labels)

def calculate_ros_events(data):
    SNOW = data['SNOW']
    ACSNOW = data['acsnow']
    RAINNC = data['rainnc']
    RAIN = RAINNC - ACSNOW
    print('RAIN calc completed')

    ros_events = (RAIN > 0.254) & (SNOW > 2.54)
    print("ROS Events filtered")

    dates = ros_events['Time'].dt.date.data
    hours = ros_events['Time'].dt.hour.data
    ros_events = ros_events.assign_coords(Date=('Time', dates), Hour=('Time', hours))

    rain_sum = RAIN.sum(dim='Time') 
    print('rain_sum for Nov-Mar tallied!') 
    rain_during_ros=RAIN.where(ros_events)
    rain_ros_sum=rain_during_ros.sum(dim='Time')
    print('rain_sum_ros for ROS events done!')
    
    rain_avg=RAIN.mean(dim='Time')
    rain_ros_avg=rain_during_ros.mean(dim='Time')
    print('Average Rain, and Average Rain during ROS calculated')
    #T2_during_ros = T2.where(ros_events)
    #T2_ros_avg = T2_during_ros.mean(dim='Time')
    #print('T2_ros_avg for ROS events done')

    ros_tally = ros_events.sum(dim='Time')
    ros_events_filtered = ros_events.where(ros_events != 0).dropna(dim='Time', how='all')
    ros_counts = ros_events_filtered.count(dim='Time')
    ros_daily_counts = ros_events_filtered.groupby('Date').count(dim='Time')
    ros_days_count = ros_daily_counts.where(ros_daily_counts > 0).count(dim='Date')
    print("ROS Daily counted")

    return {
        'ros_events': ros_events,
        'ros_tally': ros_tally,
        'ros_counts': ros_counts,
        'ros_days_count': ros_days_count,
        'rain_sum': rain_sum,
        'rain_ros_sum': rain_ros_sum,
        'rain_avg': rain_avg,
        'rain_ros_avg': rain_ros_avg
        }

def calculate_vars_by_winter_season(full_data):
    time_values = full_data['Time'].values
    season_labels = get_winter_season_labels(time_values)
    unique_seasons = sorted(set(season_labels) - {None})
    print("Starting to gather temp, snow, rain... by winter season...")
    ros_start = time.time()

    ros_tally_list = []
    ros_counts_list = []
    ros_days_count_list = []
    ros_time_series_list = []
    rain_sum_list = []
    rain_ros_sum_list = []
    rain_avg_list = []
    rain_ros_avg_list = []

    for season in unique_seasons:
        time_mask = season_labels == season
        selected_times = time_values[time_mask]
        seasonal_data = full_data.sel(Time=selected_times)

        ros_result = calculate_ros_events(seasonal_data)

        ros_tally_list.append(ros_result['ros_tally'].expand_dims(season=[season]))
        ros_counts_list.append(ros_result['ros_counts'].expand_dims(season=[season]))
        ros_days_count_list.append(ros_result['ros_days_count'].expand_dims(season=[season]))
    
        rain_sum_list.append(ros_result['rain_sum'].expand_dims(season=[season]))
        rain_ros_sum_list.append(ros_result['rain_ros_sum'].expand_dims(season=[season]))
        rain_avg_list.append(ros_result['rain_avg'].expand_dims(season=[season]))
        rain_ros_avg_list.append(ros_result['rain_ros_avg'].expand_dims(season=[season]))

    combined_ds = xr.Dataset({
        'ros_tally': xr.concat(ros_tally_list, dim='season'),
        'ros_counts': xr.concat(ros_counts_list, dim='season'),
        'ros_days_count': xr.concat(ros_days_count_list, dim='season'),
        'rain_sum':  xr.concat(rain_sum_list, dim='season'),
        'rain_ros_sum': xr.concat(rain_ros_sum_list, dim='season'),
        'rain_avg': xr.concat(rain_avg_list,dim='season'),
        'rain_ros_avg':xr.concat(rain_ros_avg_list, dim='season')
        })

    ros_end = time.time()

    print('Completed ROS by winter season.')
    print("Total time for seasonal ROS calc (min):", round((ros_end - ros_start) / 60, 2))
    return combined_ds


def main():
    parser = argparse.ArgumentParser(description="Run ROS analysis for a given number of winters starting from a specific year.")
    parser.add_argument('--start-year', type=int, required=True, help='First winter year to process (e.g., 2020 for the 2020–2021 season)')
    parser.add_argument('--n-years', type=int, required=True, help='Number of winter seasons to process')
    args = parser.parse_args()

    start_year = args.start_year
    num_years = args.n_years

    #shapefile_path = "/center1/DYNDOWN/phutton5/ROS/boundaries/ne_10m_land/ne_10m_land.shp"
    #boundaries = gpd.read_file(shapefile_path)

    for i in range(num_years):
        winter_start = start_year + i
        print(f"Starting ROS analysis for winter: {winter_start}-{winter_start + 1}")

        start_time = time.time()
        print("Script started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        try:
            #step one is to run and load the years starting from the first year that you input at the start
            ds = load_Xyrs_winter_dataset(winter_start, 1)
            lat = ds['XLAT']
            lon = ds['XLONG']
            #step two is to limit it to just fix the river/water
            ds_borough = land_mask(ds, lat, lon)
            #step 3 is to run the calculations that tall hours and days of ROS
            seasonal_ros = calculate_vars_by_winter_season(ds_borough)
            #step 4 is to output into a netcdf file
            output_path = f"/center1/DYNDOWN/phutton5/ROS/All_of_AK/All_of_AK_netcdf_files/ROS_Precip_{winter_start}-{winter_start+1}.nc"
            seasonal_ros.to_netcdf(output_path)

            print(f"Saved NetCDF: {output_path}")
        except Exception as e:
            print(f"Error processing winter {winter_start}-{winter_start + 1}: {e}")

        end_time = time.time()
        print(f"Finished winter {winter_start}-{winter_start + 1} in {round((end_time - start_time) / 60, 2)} minutes\n")


if __name__ == "__main__":
    main()


