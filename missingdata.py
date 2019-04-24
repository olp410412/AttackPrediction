import copy
import csv
import time
import pandas as pd, numpy as np, os

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

class mapper:
    def _init_(self):
        #self.x = [];
        self.y = [];
        # To tell whether the encoding is allowed or not. 0 for YES, 1 for NO
        self.flag = 0;


#a-b
def Subtraction(a,b):
    c = copy.deepcopy(a)
    while len(b) > 0:
        if b[0] in c:
            c.remove(b[0])
            b.remove(b[0])
        else:
            print("Error: b is not a sub set of a")
            break
    return c


#Get "path+filename"s from the root file
def filename(file_dir):
    L = []
    for root,dirs,files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.csv':
                L.append(os.path.join(root,file))

    return L

#Get the whole training dataset, not working yet
def combination(df1,df2):
    df_head = pd.read_csv(list[0],chunksize=100)
    for i in range(1,k):
        df_now = pd.read_csv(list[i],chunksize=100)
        df_head = df_head.append(df_now)
        #df_head = pd.concat([df_head,df_now],axis=0)
    return df_head

#Count missing value for each attribute. Put them in a same csv
def MVstastic(list):
    df = pd.read_csv('MissingValueDistribution2.csv')
    for i in range(1, len(list)):
        df0 = pd.read_csv(list[i])
        df1 = df0.isnull().sum()
        df = pd.concat([df,df1],axis=1)
        df.to_csv('MissingValueDistribution2.csv',mode = 'a')
    return 0

#Get the unique value of an Attribute(col)
def GetUnicate(col,L,addr):
    t0 = time.time()
    #Get the whole data of an Attribute(col)
    for i in range(0,len(L)):
        df = pd.read_csv(L[i], usecols=[col])
        if i == 0:
            df.to_csv(addr + '\\' + str(col) + '.csv', mode='a')
        else:
            df.to_csv(addr + '\\' + str(col)+ '.csv',mode = 'a',header=0)
    t1 = time.time()-t0
    print(t1)

    return 0

def Gettestuni(col,addr):
    t0 = time.time()
    df = pd.read_csv("data01\\test.csv",usecols=[col])
    print(df.shape)
    df.to_csv(addr + '\\' + str(col) + '.csv',mode = 'a',header=0)
    t1 = time.time()-t0
    print(t1)

    return 0

def MVfilter(df,col):
    if df[col].dtypes == object:
        df2 = df.fillna('NULL')
    else:
        df2 = df.fillna(-1)

    return df2




#Encoder of ".codes"
def LableEncode(col,addr,size):
    t0 = time.time()
    A=mapper()
    for i in range(0,size):
        #df = pd.read_csv(L[i], usecols=[col])
        df0 = pd.read_csv(addr + '\\' + str(col) + '.csv',usecols=[1])
    if df0.isnull().any().sum() > 0:
        A.flag = 1
        df1 = MVfilter(df0)
        cat = pd.Categorical(df1[col], categories=df1[col].unique(), ordered=True)
        cate = cat.codes
        A.y = cate
    else:
        #cat = pd.Categorical(df0[col],categories=df0[col].unique(),ordered=True)
        #cate = cat.codes
        #A.x = df0[col].unique()
        #A.y = cate
        A.flag = 0
    t1 = time.time()-t0
    print(t1)

    return A




if __name__=="__main__":
    L = filename('D:\\常用\\创新设计\\Detection\\temp')
    temp = {}
    _temp = {}
    #LEVL ONE
    #print (filename('D:\\常用\\创新设计\\Detection\\temp'))
    #print(L[1])
    """df = pd.read_csv(L[1])#or len(L)
    print('The dimensuion of this dataset is:'+ '\n')
    print(df.shape )
    print('\n' + 'Description:'+ '\n' )
    print(df.describe())
    print('Missing value distribution:' + '\n')
    print(df.isnull().sum())
    df2 = df.isnull().sum(axis=1)
    df2.to_csv('MissingValueDistribution_row.csv')
    #print(df.isnull().sum(axis=1))
    #MVstastic(L)"""

    #LEVEL TWO
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
    cols_to_use = Subtraction(cols, ENCODED)

    size = len(cols_to_use)
    addr = 'unique'
    DF = pd.read_csv("total_non_MV.csv",usecols = [1])
    print(DF.shape)
    for i in range(0, len(cols_to_use)):
        # Preparing
        # GetUnicate(cols[i],L,addr)
        #Gettestuni(cols[i], addr)
        #Encoding
        
        test = LableEncode(cols_to_use[i],addr,size)
        if test.flag == 1:
            print('Caution.There is missing data in the attribute:' + str(cols_to_use[i]))
            _temp.update({cols_to_use[i]:test.y})
        else:
            #temp.update({cols[i]:test.y})
            print(cols_to_use[i])
    _df = pd.DataFrame(_temp)
    _df.to_csv("total_MV.csv", index=False, sep=',')
    """df = pd.DataFrame(temp)
    df.to_csv("total_non_MV.csv",index=False,sep=',')
    
    #"""

    #LEVEL THREE

    """df0 = pd.read_csv('total.csv')
    count = 0
    for i in range(0,len(cols_to_use)):
        print(count)
        count+=1
        t0 = time.time()
        df = pd.read_csv(addr + '\\' + str(cols_to_use[i]) + '.csv',usecols=[1])
        print(cols_to_use[i])
        if i == 0:
            temp2 = pd.concat([df,df0],axis=1)
            temp2.to_csv('temp.csv')
            t1 = t0 - time.time()
            print(t1)
        else:
            df1 = pd.read_csv('temp.csv')
            temp2 = pd.concat([df,df1],axis=1)
            temp2.to_csv('temp.csv')
            t1 = t0 - time.time()
            print(t1)

    #dfnew = pd.merge(df1,df0,left_index=True, right_index=True, how='outer')
    #dfnew.to_csv("RESULT.csv")"""
    """count = 0
    for i in range(0,len(L)):
        DF = pd.read_csv(L[i], usecols=cols_to_use)
        for i in range(0,len(DF.index)):
            with open('total.csv') as csvfile:
                rows = csv.reader(csvfile)
                with open('test3.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    for row in rows:
                        count+=1
                        if 
                        row.append(DF.loc[0].values[0:-1])
                        writer.writerow(row)

    with open('total.csv') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            print('1')
    

    #"""

