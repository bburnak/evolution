import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib.path import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

class DNA:
    def __init__(self):
        self.mutating_genes = ['prob_reproduce',
                               'prob_rest',
                               'athleticism',
                               'graze_efficiency',
                               'max_energy',
                               'base_metabolism']

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
    # DNA_helix = DNA()
    mutating_genes = ['prob_reproduce',
                            'prob_rest',
                            'athleticism',
                            'graze_efficiency',
                            'max_energy',
                            'base_metabolism']
    df['reproduce_realization'] = np.random.random(size = (nLife,1))
    
    df['will_reproduce'] = df['reproduce_realization'] < df['prob_reproduce']
    print(df['will_reproduce'])
    nMutatingLife = len(df[df['will_reproduce']])
    for gene in mutating_genes:
        df.loc[df['will_reproduce'],'mutation'] = np.random.normal(1, scale = 0.005, size = (nMutatingLife,1))
        new_gene = df.loc[df['will_reproduce'], gene] * df.loc[df['will_reproduce'], 'mutation']
        df.loc[df['will_reproduce'], gene] = new_gene.clip(lower = 1e-8)
    print(df[mutating_genes].median())
    df = pd.concat([df,df[df['will_reproduce']]])
    return df

def calc_colors(df):
    nLife = len(df)
    df['changeColor_realization'] = np.random.random(size = (nLife,1))
    df['will_changeColor'] = df['changeColor_realization'] < df['prob_colorChage']
    nChangeColor = len(df[df['will_changeColor']])
    
    df.loc[df['will_changeColor'], ['colors']] += np.random.normal(size = (nChangeColor,1))*1e-2
    return df

# def calc_colors(df):
#     if 3 < len(df):
#         DNA_helix = DNA()
#         nRedDim = 3
#         DNA_PCA = PCA(n_components=nRedDim)
#         DNA_PCA.fit(df[DNA_helix.mutating_genes])
#         lifeRGB = DNA_PCA.transform(df[DNA_helix.mutating_genes])
#         lifeRGB_norm = (lifeRGB - np.amin(lifeRGB,axis=0))/(np.amax(lifeRGB,axis=0) - np.amin(lifeRGB,axis=0))
#         df['colors'] = lifeRGB_norm[:,]
#         print(df)
#     return df
    

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
    cancer_result_tmp = cancer_realization < nLife/max_life
    cancer_result = [x for xs in cancer_result_tmp for x in xs]
    df['will_die'] = (df['will_die']) | cancer_result
    return df

def terminate_mortals(df):
    df = df[~df['will_die']]
    return df

