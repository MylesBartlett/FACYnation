import pandas as pd
import numpy as np


def load_temp_precip_data(crop: str, season: str, country, regions: list, month_indexes):
    crop_season_country = [crop, season, country] if season != ''\
        else [crop, country]
    crop_season_country = '_'.join(crop_season_country)

    # Read in climate temperatures
    clim_temp_crop = pd.read_table(f'./Crop_data_files/clim_file/temp_climatology_{crop}.csv')
    clim_temp_crop.rename(columns={'Unnamed: 0': 'Crop_season_location'}, inplace=True)
    # Read in climate precipitation
    clim_precip_crop = pd.read_table(f'./Crop_data_files/clim_file/precip_climatology_{crop}.csv')
    clim_precip_crop.rename(columns={'Unnamed: 0': 'Crop_season_location'}, inplace=True)
    # Read in Yields
    yields = pd.read_table(f'./Crop_data_files/{crop}_median_yield_anoms.csv')
    years = None
    # Read in and add back mean temperature to get real temperature values
    temp_regions = []
    for i, region in enumerate(regions):
        maize_temp = pd.read_table(f'./Crop_data_files/{crop}_met_anoms/{crop_season_country}'
                                   f'_{region}_temp_anom_real.csv')
        maize_temp.rename(columns={'Unnamed: 0': 'Year'}, inplace=True)
        if years is None:   # we need to know which yield years we have climatology data for
            years = maize_temp['Year'].apply(str).values
        means = clim_temp_crop[clim_temp_crop['Crop_season_location']
                               == f'{crop_season_country}_{regions[i]}'].iloc[0, 1:, ]
        tmp = maize_temp.iloc[:, 1:].add(means)
        temp_regions.append(tmp)
    temp_regions = pd.concat(temp_regions, keys=regions)

    # Read in and add back mean precipitation to get real precipitation values
    pecip_regions = []
    for i, region in enumerate(regions):
        maize_precip = pd.read_table(
            f'./Crop_data_files/{crop}_met_anoms/{crop_season_country}_{region}_precip_anom_real.csv')
        maize_precip.rename(columns={'Unnamed: 0': 'Year'}, inplace=True)
        means = clim_precip_crop[clim_precip_crop['Crop_season_location']
                                 == f'{crop_season_country}_{regions[i]}'].iloc[0, 1:, ]
        tmp = maize_precip.iloc[:, 1:].add(means)
        pecip_regions.append(tmp)
    pecip_regions = pd.concat(pecip_regions, keys=regions)

    n_years = yields[years].shape[1]
    n_months = len(month_indexes)
    n_regions = len(regions)

    d_yields = yields[yields["Region"].isin(
            [f'{crop_season_country}_{region}' for region in regions])][years]
    missing_year_inds = d_yields.isna().any().nonzero()[0]
    n_years -= len(missing_year_inds)
    d_yields.dropna(axis=1, inplace=True)

    # Drop years with missing yield values
    d_temp = np.array(temp_regions.iloc[:, month_indexes]).reshape(
            n_regions, -1, n_months).astype(float)
    d_precip = np.array(pecip_regions.iloc[:, month_indexes]).reshape(
        n_regions, -1, n_months).astype(float)
    d_temp = np.delete(d_temp, missing_year_inds, axis=1)
    d_precip = np.delete(d_precip, missing_year_inds, axis=1)

    data = {
        'n_regions': n_regions,
        'n_years': n_years,
        'n_months': n_months,
        'd_temp': d_temp,
        'd_precip': d_precip,
        'd_yields': np.array(d_yields).astype(float) + 9.75,   # adjust for current values (this adjustment factor only applies to USA Maize)
        'n_gf': 40,
        'temp': np.arange(0, 40, 1),
        'precip': np.arange(0, 200, 5),
    }

    return data


def batch_data(data):
    x_cat = np.concatenate((data['d_temp'], data['d_precip']), -1)
    x_annual = x_cat.reshape(data['n_years'] * data['n_regions'], -1)

    batched_data = {
        'X': x_annual,
        'y': np.array(data['d_yields']).flatten()
    }

    return batched_data


def extract_data_by_year_index(data, indexes):
    data['n_years'] = len(indexes)
    temporal_data = ['d_temp', 'd_precip', 'd_yields']
    new_data = {key: value[:, indexes] if key in temporal_data else value for key, value in data.items()}
    return new_data
