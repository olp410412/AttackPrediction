import pandas as pd, numpy as np, os, gc
import time
import math

import csv

if __name__ == '__main__':
    t = time.time()
    dtypes = {

        'MachineIdentifier': 'category',
        'ProductName': 'category',
        'EngineVersion': 'category',
        'AppVersion': 'category',
        'AvSigVersion': 'category',
        'IsBeta': 'category',
        'RtpStateBitfield': 'float16',
        'IsSxsPassiveMode': 'int8',
        'DefaultBrowsersIdentifier': 'float16',
        'AVProductStatesIdentifier': 'float32',
        'AVProductsInstalled': 'float16',
        'AVProductsEnabled': 'float16',
        'HasTpm': 'int8',
        'CountryIdentifier': 'category',
        'CityIdentifier': 'category',
        'OrganizationIdentifier': 'float16',
        'GeoNameIdentifier': 'float16',
        'LocaleEnglishNameIdentifier': 'category',
        'Platform': 'category',
        'Processor': 'category',
        'OsVer': 'category',
        'OsBuild': 'int16',
        'OsSuite': 'int16',
        'OsPlatformSubRelease': 'category',
        'OsBuildLab': 'category',
        'SkuEdition': 'category',
        'IsProtected': 'float16',
        'AutoSampleOptIn': 'int8',
        'PuaMode': 'category',
        'SMode': 'float16',
        'IeVerIdentifier': 'float16',
        'SmartScreen': 'category',
        'Firewall': 'float16',
        'UacLuaenable': 'float32',
        'Census_MDC2FormFactor': 'category',
        'Census_DeviceFamily': 'category',
        'Census_OEMNameIdentifier': 'float16',
        'Census_OEMModelIdentifier': 'float32',
        'Census_ProcessorCoreCount': 'float16',
        'Census_ProcessorManufacturerIdentifier': 'float16',
        'Census_ProcessorModelIdentifier': 'float16',
        'Census_ProcessorClass': 'category',
        'Census_PrimaryDiskTotalCapacity': 'float32',
        'Census_PrimaryDiskTypeName': 'category',
        'Census_SystemVolumeTotalCapacity': 'float32',
        'Census_HasOpticalDiskDrive': 'int8',
        'Census_TotalPhysicalRAM': 'float32',
        'Census_ChassisTypeName': 'category',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches': 'float16',
        'Census_InternalPrimaryDisplayResolutionHorizontal': 'float16',
        'Census_InternalPrimaryDisplayResolutionVertical': 'float16',
        'Census_PowerPlatformRoleName': 'category',
        'Census_InternalBatteryType': 'category',
        'Census_InternalBatteryNumberOfCharges': 'float32',
        'Census_OSVersion': 'category',
        'Census_OSArchitecture': 'category',
        'Census_OSBranch': 'category',
        'Census_OSBuildNumber': 'int16',
        'Census_OSBuildRevision': 'int32',
        'Census_OSEdition': 'category',
        'Census_OSSkuName': 'category',
        'Census_OSInstallTypeName': 'category',
        'Census_OSInstallLanguageIdentifier': 'float16',
        'Census_OSUILocaleIdentifier': 'int16',
        'Census_OSWUAutoUpdateOptionsName': 'category',
        'Census_IsPortableOperatingSystem': 'int8',
        'Census_GenuineStateName': 'category',
        'Census_ActivationChannel': 'category',
        'Census_IsFlightingInternal': 'float16',
        'Census_IsFlightsDisabled': 'float16',
        'Census_FlightRing': 'category',
        'Census_ThresholdOptIn': 'float16',
        'Census_FirmwareManufacturerIdentifier': 'float16',
        'Census_FirmwareVersionIdentifier': 'float32',
        'Census_IsSecureBootEnabled': 'int8',
        'Census_IsWIMBootEnabled': 'float16',
        'Census_IsVirtualDevice': 'float16',
        'Census_IsTouchEnabled': 'int8',
        'Census_IsPenCapable': 'int8',
        'Census_IsAlwaysOnAlwaysConnectedCapable': 'float16',
        'Wdft_IsGamer': 'float16',
        'Wdft_RegionIdentifier': 'float16',
        'HasDetections': 'int8'

    }

    t0 = time.time()
    df_train = pd.read_csv('F:\\大三下\\人工智能导论\\大作业\\恶意设备预测\\train0.csv', usecols=dtypes.keys(), dtype=dtypes)
    t1 = time.time()
    print('Loaded ' + str(len(df_train)) + ' rows of data')
    print('用时' + str(t1 - t0) + 'sec' + '  共计:' + str(t1 - t) + 'sec')
    stats = []

    FE = ['EngineVersion', 'AppVersion', 'AvSigVersion', 'Census_OSVersion']
    # LOAD AND ONE-HOT-ENCODE
    OHE = ['AVProductStatesIdentifier',
           'AVProductsInstalled',
           'CountryIdentifier',
           'CityIdentifier',
           'GeoNameIdentifier',
           'LocaleEnglishNameIdentifier',
           'OsBuild',
           'OsSuite',
           'SmartScreen',
           'Census_MDC2FormFactor',
           'Census_OEMNameIdentifier',
           'Census_ProcessorCoreCount',
           'Census_ProcessorModelIdentifier',
           'Census_PrimaryDiskTotalCapacity',
           'Census_PrimaryDiskTypeName',
           'Census_TotalPhysicalRAM',
           'Census_ChassisTypeName',
           'Census_InternalPrimaryDiagonalDisplaySizeInInches',
           'Census_InternalPrimaryDisplayResolutionHorizontal',
           'Census_InternalPrimaryDisplayResolutionVertical',
           'Census_PowerPlatformRoleName',
           'Census_InternalBatteryType',
           'Census_InternalBatteryNumberOfCharges',
           'Census_OSEdition',
           'Census_OSInstallLanguageIdentifier',
           'Census_GenuineStateName',
           'Census_ActivationChannel',
           'Census_FirmwareManufacturerIdentifier',
           'Census_IsTouchEnabled',
           'Wdft_IsGamer',
           'Wdft_RegionIdentifier']

    cols = FE + OHE

    buf = df_train[cols[1]].unique()

    buf2 = df_train[cols[2]].unique()

    print(buf.T)
    print(buf.size)
    print(buf.codes)
    print("helloworld\n")
    print(buf[1])


    buffer = [-1]*1800000
    t_1 = time.time()



    col1 = df_train[cols[1]]
    counter = 0

    i = 0
    j = 0
    while (i < 1800000):
        j = 0
        while (j < 99):
            if col1[i] == buf[j]:
                buffer[i] = j
                j = 99
            j = j + 1
        i = i + 1






    print(buf)
    print(counter)

    ter_time = time.time() - t0
    print(ter_time)
