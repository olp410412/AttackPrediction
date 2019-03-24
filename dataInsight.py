import pandas as pd, numpy as np, os, gc
import time
import math

from keras import callbacks
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam

def encode_FE(df,col,verbose=1):
    d = df[col].value_counts(dropna=False)
    n = col+"_FE"
    df[n] = df[col].map(d)/d.max()
    if verbose==1:
        print('FE encoded',col)
    return [n]

def nan_check(x):
    if isinstance(x,float):
        if math.isnan(x):
            return True
    return False

# ONE-HOT-ENCODE ALL CATEGORY VALUES THAT COMPRISE MORE THAN
# "FILTER" PERCENT OF TOTAL DATA AND HAS SIGNIFICANCE GREATER THAN "ZVALUE"
def encode_OHE(df, col, filter, zvalue, tar='HasDetections', m=0.5, verbose=1):
    cv = df[col].value_counts(dropna=False)
    cvd = cv.to_dict()
    vals = len(cv)
    th = filter * len(df)
    sd = zvalue * 0.5/ math.sqrt(th)
    #print(sd)
    n = []; ct = 0; d = {}
    for x in cv.index:
        try:
            if cv[x]<th: break
            sd = zvalue * 0.5/ math.sqrt(cv[x])
        except:
            if cvd[x]<th: break
            sd = zvalue * 0.5/ math.sqrt(cvd[x])
        if nan_check(x): r = df[df[col].isna()][tar].mean()
        else: r = df[df[col]==x][tar].mean()
        if abs(r-m)>sd:
            nm = col+'_BE_'+str(x)
            if nan_check(x): df[nm] = (df[col].isna()).astype('int8')
            else: df[nm] = (df[col]==x).astype('int8')
            n.append(nm)
            d[x] = 1
        ct += 1
        if (ct+1)>=vals: break
    if verbose==1:
        print('OHE encoded',col,'- Created',len(d),'booleans')
    return [n,d]

# ONE-HOT-ENCODING from dictionary
def encode_OHE_test(df,col,dt):
    n = []
    for x in dt:
        n += encode_BE(df,col,x)
    return n

# BOOLEAN ENCODING
def encode_BE(df,col,val):
    n = col+"_BE_"+str(val)
    if nan_check(val):
        df[n] = df[col].isna()
    else:
        df[n] = df[col]==val
    df[n] = df[n].astype('int8')
    return [n]


class printAUC(callbacks.Callback):
    def __init__(self, X_train, y_train):
        super(printAUC, self).__init__()
        self.bestAUC = 0
        self.X_train = X_train
        self.y_train = y_train

    def on_epoch_end(self, epoch, logs={}):
        pred = self.model.predict(np.array(self.X_train))
        auc = roc_auc_score(self.y_train, pred)
        print("Train AUC: " + str(auc))
        pred = self.model.predict(self.validation_data[0])
        auc = roc_auc_score(self.validation_data[1], pred)
        print("Validation AUC: " + str(auc))
        if (self.bestAUC < auc):
            self.bestAUC = auc
            self.model.save("bestNet.h5", overwrite=True)
        return

# 取相同值很多的属性，空值很多的属性找出来，若取相同值很多的属性的异常值与是否被检测出，有较强相关性，仍保留
def uselessColumn():
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
    df_train = pd.read_csv('F:\\大三下\\人工智能导论\\大作业\\恶意设备预测\\train.csv', usecols=dtypes.keys(), dtype=dtypes)
    t1 = time.time()
    print('Loaded ' + str(len(df_train)) + ' rows of data')
    print('用时' + str(t1 - t0) + 'sec' + '  共计:' + str(t1 - t) + 'sec')
    stats = []
    for col in df_train.columns:
        stats.append((col, df_train[col].nunique(), df_train[col].isnull().sum() * 100 / df_train.shape[0],
                      df_train[col].value_counts(normalize=True, dropna=False).values[0] * 100, df_train[col].dtype))

    stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values',
                                            'Percentage of values in the biggest category', 'type'])

    stats_df.sort_values('Percentage of missing values', ascending=False)

    good_cols = list(df_train.columns)
    for col in df_train.columns:
        rate = df_train[col].value_counts(normalize=True, dropna=False).values[0]
        rate1 = df_train[col].isnull().sum() / df_train.shape[0]
        if rate > 0.9 or rate1 > 0.9:
            good_cols.remove(col)
       
    train1 = df_train[good_cols]
    print('用时计 ' + str(time.time() - t))
    print(good_cols)
    return [df_train, good_cols]

def function_2(good_cols):
    # LOAD AND FREQUENCY-ENCODE
    FE = ['EngineVersion', 'AppVersion', 'AvSigVersion', 'Census_OSVersion']
    # LOAD AND ONE-HOT-ENCODE
    OHE = [ 'AVProductStatesIdentifier',
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
    length = FE.__len__() + OHE.__len__()
    length1 = 0;
    same = []
    notsame = []
    for col in FE + OHE :
        if (col in good_cols):
            length1 = length1 + 1
            same.append(col)
        else:
            notsame.append(col)

    rate = length1 / length

    return [rate, same, notsame]


def function_3(df_train, good_cols):
    SampleSize = 1800000
    df_train = df_train.sample(SampleSize)
    print('Only using ' + str(SampleSize) + ' to train and validate')
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

    cols = []
    dd = []


    for x in FE:
        cols += encode_FE(df_train, x)
    for x in OHE:
        tmp = encode_OHE(df_train, x, 0.005, 5)
        cols += tmp[0];
        dd.append(tmp[1])
    print('Encoded', len(cols), 'new variables')

    for x in FE + OHE:
        del df_train[x]
    print('Removed original', len(FE + OHE), 'variables')
    x = gc.collect()

    X_train, X_val, Y_train, Y_val = train_test_split(
        df_train[cols], df_train['HasDetections'], test_size=0.3)

    # BUILD MODEL
    model = Sequential()
    model.add(Dense(100, input_dim=len(cols)))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(lr=0.01), loss="binary_crossentropy", metrics=["accuracy"])
    annealer = LearningRateScheduler(lambda x: 1e-2 * 0.95 ** x)

    # TRAIN MODEL
    model.fit(X_train, Y_train, batch_size=32, epochs=10, callbacks=[annealer, printAUC(X_train, Y_train)],
              validation_data=(X_val, Y_val), verbose=2)

  #  return

if __name__ == '__main__':
    print('Program execution')
    [train, column] = uselessColumn()
   # [rate, same, notsame] = function_2(column)
    #print(rate)
    #print(same)
    #print(notsame)
    function_3(train, column)


    print('Program end')
