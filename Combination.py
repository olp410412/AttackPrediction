import missingdata as md
import copy
import csv
import time
import pandas as pd, numpy as np, os
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import matplotlib.pyplot as plt
from pandas.api.types import is_string_dtype

ENCODED = ['EngineVersion',
           'AppVersion',
           'AvSigVersion',
           'Census_OSVersion',
           'CountryIdentifier',
           'LocaleEnglishNameIdentifier',
           'OsBuild',
           'OsSuite',
           'Census_MDC2FormFactor',
           'Census_ActivationChannel',
           'Census_IsTouchEnabled',]

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


def SKlable(col,addr):
    t0 = time.time()
    A = md.mapper()
    le = LabelEncoder()
    df0 = pd.read_csv(addr + '\\' + str(col) + '.csv', usecols=[1])
    if df0.isnull().any().sum() > 0:
        A.flag = 1
        print("Caution.There is missing data in the attribute:" + str(col))
        df1 = md.MVfilter(df0,col)
        le.fit(np.unique(df1.values))
        cate = df1.apply(le.transform)
        # A.x = df0[col].unique()
        cat = cate[col].values.tolist()
        A.y = cat
        # df1 = MVfilter(df0)
        # cat = pd.Categorical(df1[col], categories=df1[col].unique(), ordered=True)
        # cate = cat.codes
        # A.y = cate
    else:
        print(col)
        le.fit(np.unique(df0.values))
        cate = df0.apply(le.transform)
        # A.x = df0[col].unique()
        cat = cate[col].values.tolist()
        A.y = cat
        A.flag = 0
    t1 = time.time() - t0
    print(t1)
    return A

def OneHotEncoding(col,addr):
    t0 = time.time()
    A = md.mapper()
    OHE = preprocessing,OneHotEncoding()
    df0 = pd.read_csv(addr + '\\' + str(col) + '.csv', usecols=[1])
    if df0.isnull().any().sum() > 0:
        A.flag = 1
        print("Caution.There is missing data in the attribute:" + str(col))
        df1 = md.MVfilter(df0, col)
        OHE.fit(np.unique(df1.values))
        cate = df1.apply(OHE.transform)
        # A.x = df0[col].unique()
        cat = cate[col].values.tolist()
        A.y = cat
        # df1 = MVfilter(df0)
        # cat = pd.Categorical(df1[col], categories=df1[col].unique(), ordered=True)
        # cate = cat.codes
        # A.y = cate
    else:
        print(col)
        OHE.fit(np.unique(df0.values))
        cate = df0.apply(OHE.transform)
        # A.x = df0[col].unique()
        cat = cate[col].values.tolist()
        A.y = cat
        A.flag = 0
    t1 = time.time() - t0
    print(t1)
    return A

def combination(file1,file2,file_new):
    c1 = 0
    c2 = 0
    with open(file1) as csvfile1:
        rows1 = csv.reader(csvfile1)
        for row in rows1:
            c1+=1
            print(c1)
            with open(file2) as csvfile2:
                rows2 = csv.reader(csvfile2)
                for row2 in rows2:
                    c2+=1
                    if c2 == c1:
                        list2 = row2 + row
                        print(list2)
                        with open(file_new, 'a', newline='') as csvfile3:
                            writer = csv.writer(csvfile3)
                            writer.writerow(list2)
            c2 = 0
    return 0

def combination2():
    print("space")

"""
    
    
                     
               
"""

if __name__=="__main__":
    addr = 'unique'
    cols = FE + OHE
    counter = 0
    """for i in range(0,len(cols)):
        print("【" + str(counter) + "】 " + cols[i])
        df = pd.read_csv(addr + "\\" + cols[i] + ".csv",usecols=[1])
        print(df[cols[i]].dtypes)
        if df.isnull().any().sum() == 0:
            uni = np.unique(df.values)
            print(uni)
            print(len(uni))
        else:
            df0=md.MVfilter(df,cols[i])
            uni = np.unique(df0.values)
            print(uni)
            print(len(uni))
        try:
            if df[cols[i]].dtypes == "int64":
                #df.astype(float)
                df.plot(kind='line',figsize=[5,5],legend=True,title=cols[i])
                plt.show
            elif df[cols[i]].dtypes == "float64":
                df.plot(kind='line', figsize=[5, 5], legend=True, title=cols[i])
                plt.show
        except:
            print("out of memory")

        counter+=1#"""
    _temp = {}
    cols_to_use = md.Subtraction(cols, ENCODED)
    for i in range(13,len(cols_to_use)):
        test = SKlable(cols_to_use[i],addr)
        if test.flag == 1:
            _temp.update({cols_to_use[i]:test.y})

    df = pd.DataFrame(_temp)
    df.to_csv("SKLEARN\\LableEncode_MV2.csv", index=False, sep=',')

    """df = pd.DataFrame(temp)
    df.to_csv("SKLEARN\\LableEncode_nonMV7.csv", index=False, sep=',')

    #"""
