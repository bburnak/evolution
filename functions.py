import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib.path import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

class DNA:
    def __init__(self):
        self.mutating_genes = ['prob_reproduce',
                               'prob_rest',
                               'athleticism',
                               'graze_efficiency',
                               'max_energy',
                               'base_metabolism',
                               'reproduce_efficiency',
                               'range_perception',
                               'cancer_immunity']

def calc_loc(df):
    nLife = len(df)
    df['state_realization_move'] = np.random.random(size = (nLife,1))
    df['resting'] = df['state_realization_move'] < df['prob_rest']
    df['xdir'] = np.random.normal(0, size = (nLife,1))*1e-1
    df['ydir'] = np.random.normal(0, size = (nLife,1))*1e-1
    
    df.loc[~df['resting'],'xloc'] = df.loc[~df['resting'],'xloc'] + df.loc[~df['resting'],'xdir']
    df.loc[~df['resting'],'yloc'] = df.loc[~df['resting'],'yloc'] + df.loc[~df['resting'],'ydir']
    df = enervate(df)
    return df

def calc_reproduction(df):
    nLife = len(df)
    DNA_helix = DNA()
    df['reproduce_realization'] = np.random.random(size = (nLife,1))
    df['will_reproduce'] = df['reproduce_realization'] < df['prob_reproduce']
    df.loc[df['will_reproduce'], 'energy'] = df.loc[df['will_reproduce'], 'energy']/2 * df.loc[df['will_reproduce'], 'reproduce_efficiency']
    nMutatingLife = len(df[df['will_reproduce']])
    for gene in DNA_helix.mutating_genes:
        df.loc[df['will_reproduce'],'mutation'] = np.random.normal(1, scale = 0.005, size = (nMutatingLife,1))
        new_gene = df.loc[df['will_reproduce'], gene] * df.loc[df['will_reproduce'], 'mutation']
        df.loc[df['will_reproduce'], gene] = new_gene.clip(lower = 1e-8)
    print(df[DNA_helix.mutating_genes].median())
    print(df['energy'])
    df = pd.concat([df,df[df['will_reproduce']]])
    return df

# def calc_colors(df):
#     nLife = len(df)
#     df['changeColor_realization'] = np.random.random(size = (nLife,1))
#     df['will_changeColor'] = df['changeColor_realization'] < df['prob_colorChage']
#     nChangeColor = len(df[df['will_changeColor']])
    
#     df.loc[df['will_changeColor'], ['colors']] += np.random.normal(size = (nChangeColor,1))*1e-2
#     return df

def calc_colors(df):
    if 3 < len(df):
        DNA_helix = DNA()
        nRedDim = 3
        DNA_PCA = PCA(n_components=nRedDim)
        DNA_PCA.fit(df[DNA_helix.mutating_genes])
        lifeRGB = DNA_PCA.transform(df[DNA_helix.mutating_genes])
        lifeRGB_norm = (lifeRGB - np.amin(lifeRGB,axis=0))/(np.amax(lifeRGB,axis=0) - np.amin(lifeRGB,axis=0))
        # df['colors'] = lifeRGB_norm[:,]
        # print(lifeRGB_norm[:,0])
        # print(lifeRGB_norm[:,1])
        # print(lifeRGB_norm[:,2])
        df['red'] = lifeRGB_norm[:,0]
        df['green'] = lifeRGB_norm[:,1]
        df['blue'] = lifeRGB_norm[:,2]
        # for i in range(len(df)):
        #     print(df.iloc[i,'colors'])
    return df
    

def kill_outsiders(df):
    outsiders = (df['xloc']<-10) | (10<df['xloc']) | (df['yloc']<-10) | (10<df['yloc'])
    df = df[~outsiders]
    return df

def enervate(df):
    dir_total = np.array(df.loc[~df['resting'],'xdir']) + np.array(df.loc[~df['resting'],'ydir'])
    df.loc[~df['resting'],'energy_spent'] = abs(dir_total*(1 - df.loc[~df['resting'],'athleticism']) )
    df.loc[df['resting'],'energy_spent'] = df.loc[df['resting'],'base_metabolism']*1e-4
    df['energy'] -= df['energy_spent']
    df['energy_ratio'] = df['energy'] / df['max_energy']
    df['will_die'] = (df['will_die']) | (df['energy'] < 0)
    return df

def get_cancer(df):
    max_life = 100000
    nLife = len(df)
    cancer_realization = np.random.random(size = (nLife,1))
    df['tmp'] = cancer_realization
    cancer_result = (df['tmp'] < nLife/max_life)*(1-df['cancer_immunity'])
    df['will_die'] = (df['will_die']) | (cancer_result)
    return df

def get_nearest_life(df):
    nLife = len(df)
    if nLife > 3:
        life_dist = squareform(pdist(df[['xloc','yloc']]))
    return life_dist



def terminate_mortals(df):
    df = df[~df['will_die']]
    return df

