import numpy as np
import pandas as pd
import sys
import xlsxwriter

def dropColumns(data, drop):
    # drop columns which are not needed
    for col in drop:
        data.drop(col, 1, inplace=True)


def getMean(data, col, cond_value):
    masked = np.ma.masked_array(data[col], data[col] == cond_value)
    mean = masked.mean()
    return mean


colnames_cloud_type = ['STATIONS_ID', 'MESS_DATUM', 'QN_8', 'V_N', 'V_N_I', 'V_S1_CS', 'V_S1_CSA', 'V_S1_HHS',
                       'V_S1_NS', 'V_S2_CS', 'V_S2_CSA', 'V_S2_HHS', 'V_S2_NS', 'V_S3_CS', 'V_S3_CSA', 'V_S3_HHS',
                       'V_S3_NS', 'V_S4_CS', 'V_S4_CSA', 'V_S4_HHS', 'V_S4_NS', 'eor']
drop_cloud_type = ['QN_8', 'V_N_I', 'V_S1_CSA', 'V_S2_CSA', 'V_S3_CS', 'V_S3_CSA', 'V_S3_HHS', 'V_S3_NS', 'V_S4_CS',
                   'V_S4_CSA',
                   'V_S4_HHS', 'V_S4_NS', 'eor']

colnames_cloudiness = ['STATIONS_ID', 'MESS_DATUM', 'QN_8', 'V_N_I', 'V_N', 'eor']
drop_cloudiness = ['QN_8', 'V_N_I', 'eor']

colnames_precipitation = ['STATIONS_ID', 'MESS_DATUM', 'QN_8', 'R1', 'RS_IND', 'WRTR', 'eor']
drop_precipitation = ['QN_8', 'WRTR', 'eor']

colnames_pressure = ['STATIONS_ID', 'MESS_DATUM', 'QN_8', 'P', 'P0', 'eor']
drop_pressure = ['QN_8', 'eor']

colnames_sun = ['STATIONS_ID', 'MESS_DATUM', 'QN_7', 'SD_SO', 'eor']
drop_sun = ['QN_7', 'eor']

colnames_temperature = ['STATIONS_ID', 'MESS_DATUM', 'QN_9', 'TT_TU', 'RF_TU', 'eor']
drop_temperature = ['QN_9', 'eor']

colnames_wind = ['STATIONS_ID', 'MESS_DATUM', 'QN_3', 'F', 'D', 'eor']
drop_wind = ['QN_3', 'eor']

cloud_type = pd.read_csv('Weather/cloud_type/cloud_type.txt', names=colnames_cloud_type, sep=';', header=1)
cloudiness = pd.read_csv('Weather/cloudiness/cloudiness.txt', names=colnames_cloudiness, sep=';', header=1)
precipitation = pd.read_csv('Weather/precipitation/precipitation.txt', names=colnames_precipitation, sep=';', header=1)
pressure = pd.read_csv('Weather/pressure/pressure.txt', names=colnames_pressure, sep=';', header=1)
sun = pd.read_csv('Weather/sun/sun.txt', names=colnames_sun, sep=';', header=1)
temperature = pd.read_csv('Weather/temperature/temperature.txt', names=colnames_temperature, sep=';', header=1)
wind = pd.read_csv('Weather/wind/wind.txt', names=colnames_wind, sep=';', header=1)

dropColumns(cloud_type, drop_cloud_type)
dropColumns(cloudiness, drop_cloudiness)
dropColumns(precipitation, drop_precipitation)
dropColumns(pressure, drop_pressure)
dropColumns(sun, drop_sun)
dropColumns(temperature, drop_temperature)
dropColumns(wind, drop_wind)

cloud_type['V_S1_HHS'] = pd.to_numeric(cloud_type['V_S1_HHS'])
cloud_type['V_S1_NS'] = pd.to_numeric(cloud_type['V_S1_NS'])
cloud_type['V_S1_CS'] = pd.to_numeric(cloud_type['V_S1_CS'])
cloud_type['V_S2_HHS'] = pd.to_numeric(cloud_type['V_S2_HHS'])
cloud_type['V_S2_CS'] = pd.to_numeric(cloud_type['V_S2_CS'])
pressure['P'] = pd.to_numeric(pressure['P'])
pressure['P0'] = pd.to_numeric(pressure['P0'])
sun['SD_SO'] = pd.to_numeric(sun['SD_SO'])
temperature['TT_TU'] = pd.to_numeric(temperature['TT_TU'])
temperature['RF_TU'] = pd.to_numeric(temperature['RF_TU'])
wind['D'] = pd.to_numeric(wind['D'])
wind['F'] = pd.to_numeric(wind['F'])


mean_V_S1_HHS = getMean(cloud_type, 'V_S1_HHS', -999)
mean_V_S1_NS = getMean(cloud_type, 'V_S1_NS', -999)
mean_V_S1_CS = getMean(cloud_type, 'V_S1_CS', -999)
mean_V_S2_HHS = getMean(cloud_type, 'V_S2_HHS', -999)
mean_V_S2_NS = getMean(cloud_type, 'V_S2_NS', -999)
mean_V_S2_CS = getMean(cloud_type, 'V_S2_CS', -999)
mean_P = getMean(pressure, 'P', -999)
mean_P0 = getMean(pressure, 'P0', -999)
mean_SD_SO = getMean(sun, 'SD_SO', -999)
mean_TT_TU = getMean(temperature, 'TT_TU', -999)
mean_RF_TU = getMean(temperature, 'RF_TU', -999)
mean_D = getMean(wind, 'D', -999)
mean_F = getMean(wind, 'F', -999)


cloud_type["V_S1_HHS"] = np.where(cloud_type["V_S1_HHS"] == -999, int(mean_V_S1_HHS), cloud_type["V_S1_HHS"])
cloud_type["V_S1_NS"] = np.where(cloud_type["V_S1_NS"] == -999, int(mean_V_S1_NS), cloud_type["V_S1_NS"])
cloud_type["V_S1_CS"] = np.where(cloud_type["V_S1_CS"] == -999, int(mean_V_S1_CS), cloud_type["V_S1_CS"])
cloud_type["V_S2_HHS"] = np.where(cloud_type["V_S2_HHS"] == -999, int(mean_V_S2_HHS), cloud_type["V_S2_HHS"])
cloud_type["V_S2_NS"] = np.where(cloud_type["V_S2_NS"] == -999, int(mean_V_S2_NS), cloud_type["V_S2_NS"])
cloud_type["V_S2_CS"] = np.where(cloud_type["V_S2_CS"] == -999, int(mean_V_S2_CS), cloud_type["V_S2_CS"])
pressure["P"] = np.where(pressure["P"] == -999, mean_P, pressure["P"])
pressure["P0"] = np.where(pressure["P0"] == -999, mean_P0, pressure["P0"])
sun["SD_SO"] = np.where(sun["SD_SO"] == -999, mean_SD_SO, sun["SD_SO"])
temperature["TT_TU"] = np.where(temperature["TT_TU"] == -999, mean_TT_TU, temperature["TT_TU"])
temperature["RF_TU"] = np.where(temperature["RF_TU"] == -999, mean_RF_TU, temperature["RF_TU"])
precipitation["R1"] = np.where(precipitation["R1"] == -999, 0, precipitation["R1"])
precipitation["RS_IND"] = np.where(precipitation["RS_IND"] == -999, 0, precipitation["RS_IND"])
wind["D"] = np.where(wind["D"] == -999, int(mean_D), wind["D"])
wind["F"] = np.where(wind["F"] == -999, int(mean_F), wind["F"])



result = pd.merge(cloud_type, cloudiness, on=['STATIONS_ID', 'MESS_DATUM'])
result = pd.merge(result, pressure, on=['STATIONS_ID', 'MESS_DATUM'])
result = pd.merge(result, precipitation, on=['STATIONS_ID', 'MESS_DATUM'])
result = pd.merge(result, temperature, on=['STATIONS_ID', 'MESS_DATUM'])
result = pd.merge(result, sun, on=['STATIONS_ID', 'MESS_DATUM'])
result = pd.merge(result, wind, on=['STATIONS_ID', 'MESS_DATUM'])


print(result)
results = pd.DataFrame(result, columns=['STATIONS_ID', 'MESS_DATUM', 'V_N_x', 'V_S1_CS', 'V_S1_HHS',
                                        'V_S1_NS', 'V_S2_CS', 'V_S2_HHS', 'V_S2_NS', 'V_N_y', 'P', 'P0',
                                        'R1', 'RS_IND',
                                        'TT_TU', 'RF_TU', 'SD_SO', 'F', 'D']).to_csv('results_V2.csv')
