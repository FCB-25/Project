import numpy as np
import pandas as pd


def dropColumns(data, drop):
    # drop columns which are not needed
    for col in drop:
        data.drop(col, 1, inplace=True)

def getMean(data,col,cond_value):
    masked = np.ma.masked_array(data[col], data[col] == cond_value)
    mean = masked.mean()
    return mean

colnames_cloud_type = ['STATIONS_ID', 'MESS_DATUM', 'QN_8', 'V_N', 'V_N_I', 'V_S1_CS', 'V_S1_CSA', 'V_S1_HHS',
                       'V_S1_NS', 'V_S2_CS', 'V_S2_CSA', 'V_S2_HHS',
                       'V_S2_NS', 'V_S3_CS', 'V_S3_CSA', 'V_S3_HHS', 'V_S3_NS', 'V_S4_CS', 'V_S4_CSA', 'V_S4_HHS',
                       'V_S4_NS', 'eor']
drop_cloud_type = ['QN_8', 'V_S1_CSA', 'V_S2_CSA', 'V_S3_CS', 'V_S3_CSA', 'V_S3_HHS', 'V_S3_NS', 'V_S4_CS', 'V_S4_CSA',
                   'V_S4_HHS',
                   'V_S4_NS', 'eor']

colnames_cloudiness = ['STATIONS_ID', 'MESS_DATUM', 'QN_8', 'V_N_I', 'V_N', 'eor']
drop_cloudiness = ['QN_8', 'eor']

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

cloud_type = pd.read_csv('Weather/cloud_type/cloud_type.txt', names=colnames_cloud_type, sep=';',header=1)
cloudiness = pd.read_csv('Weather/cloudiness/cloudiness.txt', names=colnames_cloudiness, sep=';')
precipitation = pd.read_csv('Weather/precipitation/precipitation.txt', names=colnames_precipitation, sep=';')
pressure = pd.read_csv('Weather/pressure/pressure.txt', names=colnames_pressure, sep=';')
sun = pd.read_csv('Weather/sun/sun.txt', names=colnames_sun, sep=';')
temperature = pd.read_csv('Weather/temperature/temperature.txt', names=colnames_temperature, sep=';')
wind = pd.read_csv('Weather/wind/wind.txt', names=colnames_wind, sep=';')

dropColumns(cloud_type, drop_cloud_type)
dropColumns(cloudiness, drop_cloudiness)
dropColumns(precipitation, drop_precipitation)
dropColumns(pressure, drop_pressure)
dropColumns(sun, drop_sun)
dropColumns(temperature, drop_temperature)
dropColumns(wind, drop_wind)

#cloud_type.where(cloud_type[:,'V_S1_HHS'] == '-999',cloud_type[:,'V_S1_HHS'].median,inplace=True)

cloud_type['V_S2_HHS'] = pd.to_numeric(cloud_type['V_S2_HHS'])

#cloud_type['V_S2_HHS'] = np.where(cloud_type['V_S2_HHS'] == '-999', median,cloud_type['V_S2_HHS'])

#print(cloud_type['V_S2_HHS'])

print(getMean(cloud_type,'V_S2_HHS',-999))
