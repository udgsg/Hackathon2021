import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np 

# sort the dataframe
def stateData(dataframe, state):
    # Create a copy
    dataframe2 = dataframe.copy()
    # set the index to be this and don't drop
    dataframe2.set_index(keys=['state'], drop=False,inplace=True)
    return dataframe2.loc[dataframe2.state==state]

def correlation(file):
    data = pd.read_csv(file)
    
    # Drop empty columns and columns that only contain NaNs or zeroes
    data = data.dropna(axis=1)#.loc[:, (data != 0).any(axis=0)]

    corr = data.corr()
    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    plt.show()

def correlation_by_state(dataframe, state):    
    # Drop empty columns and columns that only contain NaNs or zeroes
    data = dataframe.dropna(axis=1).loc[:, (dataframe != 0).any(axis=0)]

    corr = data.corr()
    plt.figure()
    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    
    savename = state + '.png'
    plt.title(state)
    plt.savefig(savename)

dataframe_all = pd.read_csv(r"C:\Users\joeav\Downloads\all-states-history.csv")
# get list of states
states = set(dataframe_all['state'])

counter = 0
for state in states:
    counter += 1
    print(counter)
    dataframe_state = stateData(dataframe_all, state)
    correlation_by_state(dataframe_state, state)