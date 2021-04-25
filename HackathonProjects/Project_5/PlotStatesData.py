# sort the dataframe
def stateData(data, state):
    # Create a copy
    dataframe2 = dataframe.copy()
    # set the index to be this and don't drop
    dataframe2.set_index(keys=['state'], drop=False,inplace=True)
    return dataframe2.loc[dataframe2.state==state]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing

def scaleValues(dataframe):
    col_names = dataframe.columns.tolist()
    # Stardardize
    data = dataframe.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)
    
    dataframe2= pd.DataFrame(data_scaled)
    dataframe2.columns = col_names   
    
    return dataframe2

def reArrange(dataframe):
    # Reverse rows
    dataframe = dataframe.iloc[::-1]
    # Rename rows
    dataframe.index = range(1, dataframe.shape[0] + 1 )
    
    return dataframe

#for col in col_names:
#    dataframe2.set_index(col).plot(figsize=(30,10), title=col, fontsize=30).legend(fontsize=20)    


def plotStates(dataframe, stateNames = None):
    if stateNames is None:
        stateNames = dataframe['state'].unique().tolist()
    for stt in stateNames:
        stateDataframe = stateData(dataframe, stt)
        stateDataframe = stateDataframe.dropna(axis=1).drop(['date', 'state'], axis=1).loc[:, (stateDataframe != 0).any(axis=0)]
        stateDataframe = scaleValues(stateDataframe)
        stateDataframe = reArrange(stateDataframe)

        stateDataframe.plot(figsize=(30,10), title=stt, fontsize=30).legend(fontsize=30)
        #print(stateDataframe)
    

dataframe = pd.read_csv("all-states-history.csv")   
plotStates(dataframe, stateNames = None)
