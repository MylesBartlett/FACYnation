import pandas as pd
import numpy as np

_US_MAIZE_STATES = ['Indiana', 'Illinois', 'Ohio', 'Nebraska', 'Iowa', 'Minnesota']


def prepare_us_maize_data(month_start=3, month_end=9):

    assert month_start >= 0 and month_end >= 0
    if month_end < month_start:
        raise ValueError('Invalid month range')
    # Read in climate temperatures
    clim_temp_maize = pd.read_table('./Crop_data_files/clim_file/temp_climatology_Maize.csv')
    clim_temp_maize.rename(columns={'Unnamed: 0': 'Crop_season_location'}, inplace=True)
    # Read in climate precipitation
    clim_precip_maize = pd.read_table('./Crop_data_files/clim_file/precip_climatology_Maize.csv')
    clim_precip_maize.rename(columns={'Unnamed: 0': 'Crop_season_location'}, inplace=True)
    # Read in Yields
    yields = pd.read_table('./Crop_data_files/Maize_median_yield_anoms.csv')

    # Read in and add back mean temperature to get real temperature values
    temp_states = []
    for i, s in enumerate(_US_MAIZE_STATES):
        maize_temp = pd.read_table('./Crop_data_files/maize_met_anoms/Maize_Spring_USA_' + s + '_temp_anom_real.csv')
        maize_temp.rename(columns={'Unnamed: 0': 'Year'}, inplace=True)
        tmp = maize_temp.iloc[:, 1:].add(
            clim_temp_maize[clim_temp_maize['Crop_season_location'] == 'Maize_Spring_USA_' + _US_MAIZE_STATES[0]].iloc[0, 1:, ])
        temp_states.append(tmp)
    temp_states = pd.concat(temp_states, keys=_US_MAIZE_STATES)

    # Read in and add back mean precipitation to get real precipitation values
    precip_states = []
    for i, s in enumerate(_US_MAIZE_STATES):
        maize_precip = pd.read_table(
            './Crop_data_files/maize_met_anoms/Maize_Spring_USA_' + s + '_precip_anom_real.csv')
        maize_precip.rename(columns={'Unnamed: 0': 'Year'}, inplace=True)
        tmp = maize_precip.iloc[:, 1:].add(
            clim_precip_maize[clim_precip_maize['Crop_season_location'] == 'Maize_Spring_USA_' + _US_MAIZE_STATES[0]].iloc[0,
            1:, ])
        precip_states.append(tmp)
    precip_states = pd.concat(precip_states, keys=_US_MAIZE_STATES)

    n_years = np.array(yields[yields['Region'] == 'Maize_Spring_USA_Indiana'].iloc[0, 22:]).size
    n_months = month_end - month_start

    data = {
        'n_regions': len(_US_MAIZE_STATES),
        'n_years': n_years,
        'd_temp': np.array(temp_states.iloc[:, month_start: month_end]).reshape(
            len(_US_MAIZE_STATES), -1, n_months
        ).astype(float),
        'd_precip': np.array(precip_states.iloc[:, month_start: month_end]).reshape(
            len(_US_MAIZE_STATES), -1, n_months).astype(float),
        'd_yields': np.array(yields[yields["Region"].isin(
            [f'Maize_Spring_USA_{s}' for s in _US_MAIZE_STATES]
        )].iloc[:, 22:]).astype(float) + 6,
        'n_gf': 40,
        'temp': np.arange(0, 40, 1),
        'precip': np.arange(0, 200, 5)
    }

    return data


def extract_data_by_year_index(data, indexes):
    data['n_years'] = len(indexes)
    temporal_data = ['d_temp', 'd_precip', 'd_yields']
    new_data = {key: value[:, indexes] if key in temporal_data else value for key, value in data.items()}
    return new_data
