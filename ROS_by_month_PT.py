import os
import glob
import time
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr

def load_winter_months_dataset(start_winter_year, num_years):
    print("Loading multi-year winter-month dataset (Nov–Mar)...")
    load_start = time.time()
    all_file_list = []

    for i in range(num_years):
        year1 = start_winter_year + i
        year2 = year1 + 1

        path1 = f'/import/beegfs/CMIP6/wrf_era5/04km/{year1}'
        path2 = f'/import/beegfs/CMIP6/wrf_era5/04km/{year2}'

        months_first_year = ['11', '12']  # Nov, Dec
        months_second_year = ['01', '02', '03']  # Jan–Mar

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
    print("Time to load winter months dataset (min):", round((load_end - load_start) / 60, 2))

    def select_vars(ds):
        # include temp, T2, precipitation, snow
        return ds[['T2', 'SNOW', 'acsnow', 'rainnc', 'temp', 'XLAT', 'XLONG', 'Time']]

    if all_file_list:
        data = xr.open_mfdataset(all_file_list, combine='by_coords', preprocess=select_vars)
        print("Winter months dataset opened and combined!")
        return data
    else:
        raise ValueError("No files found for the winter months!")

def land_mask(data, lat, lon):
    os.chdir("/import/beegfs/CMIP6/wrf_era5")
    geo = xr.open_dataset("geo_em.d02.nc")
    landmask = geo['LANDMASK'].squeeze(dim="Time")
    landmask_expanded = landmask.expand_dims(Time=data.Time)

    data_fixed = data.where(landmask_expanded == 1)
    data_fixed = data_fixed.chunk(dict(south_north=-1, west_east=-1))
    data_filled = data_fixed.interpolate_na(dim="south_north", method="nearest", fill_value="extrapolate")
    data_filled = data_filled.interpolate_na(dim="west_east", method="nearest", fill_value="extrapolate")
    data_final = data_filled.where(~np.isnan(landmask_expanded))

    oceanmask = geo['LU_INDEX'].squeeze(dim='Time')
    oceanmask_expanded = oceanmask.expand_dims(Time=data.Time)
    data_masked = data_final.where(oceanmask_expanded != 17)

    print("Land Mask and Ocean Mask applied.")
    return data_masked

def get_season_month_labels(time_index):
    labels, months = [], []
    for t in pd.to_datetime(time_index):
        if t.month in [11, 12]:
            season = f"{t.year}-{t.year + 1}"
        elif t.month in [1, 2, 3]:
            season = f"{t.year - 1}-{t.year}"
        else:
            season = None
        if season:
            labels.append(season)
            months.append(f"{t.month:02d}")
        else:
            labels.append(None)
            months.append(None)
    return np.array(labels), np.array(months)

def calculate_ros_events(data):
    T2 = data['T2']
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
    #saving T2
    T2_avg = T2.mean(dim='Time')
    T2_during_ros = T2.where(ros_events)
    T2_ros_avg = T2_during_ros.mean(dim='Time')
    print('T2_avg and T2_ros_avg done')
    #saving temps 
    if 'temp' in data:
        try:
            selected_t_levels = data['temp'].sel(interp_level=[850., 925., 950.])
            temp_levels_avg = selected_t_levels.mean(dim='Time')
            temp_levels_ros_avg = selected_t_levels.where(ros_events).mean(dim='Time')
            print('Pressure-level temp avgs done')
        except Exception as e:
            print(f'Error selecting pressure levels: {e}')
            temp_levels_avg = None
            temp_levels_ros_avg = None
    else:
        temp_levels_avg = None
        temp_levels_ros_avg = None
        print('No pressure-level temperature data found')

    # saving rain and snow..
    rain_sum = RAIN.sum(dim='Time')
    rain_during_ros = RAIN.where(ros_events)
    rain_ros_sum = rain_during_ros.sum(dim='Time')

    rain_avg = RAIN.mean(dim='Time')
    rain_ros_avg = rain_during_ros.mean(dim='Time')

    snow_avg = SNOW.mean(dim='Time')
    snow_ros_avg = SNOW.where(ros_events).mean(dim='Time')

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
        'rain_ros_avg': rain_ros_avg,
        'swe_avg': snow_avg,
        'swe_ros_avg': snow_ros_avg,
        'T2_avg': T2_avg,
        'T2_ros_avg': T2_ros_avg,
        'temp_levels_avg': temp_levels_avg,
        'temp_levels_ros_avg': temp_levels_ros_avg
    }

def calculate_vars_by_season_month(full_data):
    time_values = full_data['Time'].values
    season_labels, month_labels = get_season_month_labels(time_values)
    unique_pairs = sorted(set(zip(season_labels, month_labels)) - {(None, None)})

    print("Starting ROS + Temp calculations by month within each season...")
    ros_start = time.time()

    result_vars = {
        'ros_tally': [],
        'ros_counts': [],
        'ros_days_count': [],
        'rain_sum': [],
        'rain_ros_sum': [],
        'rain_avg': [],
        'rain_ros_avg': [],
        'swe_avg': [],
        'swe_ros_avg': [],
        'T2_avg': [],
        'T2_ros_avg': [],
        'temp_levels_avg': [],
        'temp_levels_ros_avg': []
    }

    for season, month in unique_pairs:
        time_mask = (season_labels == season) & (month_labels == month)
        selected_times = time_values[time_mask]
        monthly_data = full_data.sel(Time=selected_times)

        print(f"Processing {season} - Month {month}")
        ros_result = calculate_ros_events(monthly_data)

        for key in result_vars:
            if ros_result[key] is not None:
                result_vars[key].append(ros_result[key].expand_dims(season=[season], month=[month]))

    combined_ds = xr.Dataset({
        key: xr.concat(result_vars[key], dim='month') for key in result_vars if result_vars[key]
    })

    ros_end = time.time()
    print("Completed ROS monthly calculations by season.")
    print("Total time (min):", round((ros_end - ros_start) / 60, 2))
    return combined_ds

def main():
    parser = argparse.ArgumentParser(description="Run ROS and temperature analysis per month within each winter season.")
    parser.add_argument('--start-year', type=int, required=True)
    parser.add_argument('--n-years', type=int, required=True)
    args = parser.parse_args()

    start_year = args.start_year
    num_years = args.n_years

    for i in range(num_years):
        winter_start = start_year + i
        print(f"Starting ROS analysis for winter: {winter_start}-{winter_start + 1}")
        start_time = time.time()
        print("Script started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        try:
            ds = load_winter_months_dataset(winter_start, 1)
            lat = ds['XLAT']
            lon = ds['XLONG']
            ds_masked = land_mask(ds, lat, lon)
            ros_by_month = calculate_vars_by_season_month(ds_masked)

            output_path = f"/center1/DYNDOWN/phutton5/ROS/All_of_AK/All_of_AK_netcdf_files/ROS_Monthly_{winter_start}-{winter_start+1}.nc"
            ros_by_month.to_netcdf(output_path)
            print(f"Saved NetCDF: {output_path}")

        except Exception as e:
            print(f"Error processing winter {winter_start}-{winter_start + 1}: {e}")

        end_time = time.time()
        print(f"Finished {winter_start}-{winter_start + 1} !!!")

if __name__ == "__main__":
    main()

