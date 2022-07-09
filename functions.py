import random
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
                               'cancer_immunity',
                               'chase_tendency',
                               'cowardice',
                               'hunting_tendency',
                               'hunting_efficiency']

def calc_loc(df):
    nLife = len(df)
    df['state_realization_move'] = np.random.random(size = (nLife,1))
    df['resting'] = df['state_realization_move'] < df['prob_rest']
    df = chase_n_runaway(df)
    xvec = np.random.normal(0, size = (nLife,1))
    yvec = np.random.normal(0, size = (nLife,1))
    totalvec = np.sqrt(xvec**2 + yvec**2+1e-8)
    df['rand_xdir'] = xvec/totalvec*1e-1
    df['rand_ydir'] = yvec/totalvec*1e-1
    df['xdir'] = df['rand_xdir'] + df['chase_xdir'] + df['runaway_xdir']
    df['ydir'] = df['rand_ydir'] + df['chase_ydir'] + df['runaway_ydir']
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
        df.loc[df['will_reproduce'],'mutation'] = np.random.normal(1, scale = 0.01, size = (nMutatingLife,1))
        new_gene = df.loc[df['will_reproduce'], gene] * df.loc[df['will_reproduce'], 'mutation']
        df.loc[df['will_reproduce'], gene] = new_gene.clip(lower = 1e-8)
    print(df[DNA_helix.mutating_genes].median())
    # print(df['energy'])
    df = pd.concat([df,df[df['will_reproduce']]])
    return df

def calc_colors(df):
    nLife = len(df)
    df['changeColor_realization'] = np.random.random(size = (nLife,1))
    df['will_changeColor'] = df['changeColor_realization'] < df['prob_colorChage']
    nChangeColor = len(df[df['will_changeColor']])
    df.loc[df['will_changeColor'], ['colors']] += np.random.normal(size = (nChangeColor,1))*3e-2
    return df

# def calc_colors(df):
#     if 3 < len(df):
#         DNA_helix = DNA()
#         nRedDim = 3
#         DNA_PCA = PCA(n_components=nRedDim)
#         DNA_PCA.fit(df[DNA_helix.mutating_genes])
#         lifeRGB = DNA_PCA.transform(df[DNA_helix.mutating_genes])
#         lifeRGB_norm = (lifeRGB - np.amin(lifeRGB,axis=0))/(np.amax(lifeRGB,axis=0) - np.amin(lifeRGB,axis=0))
#         # df['colors'] = lifeRGB_norm[:,]
#         # print(lifeRGB_norm[:,0])
#         # print(lifeRGB_norm[:,1])
#         # print(lifeRGB_norm[:,2])
#         df['red'] = lifeRGB_norm[:,0]
#         df['green'] = lifeRGB_norm[:,1]
#         df['blue'] = lifeRGB_norm[:,2]
#         # for i in range(len(df)):
#         #     print(df.iloc[i,'colors'])
#     return df
    

def handle_outsiders(df):
    outsider_NE = (df['yloc']<-10) | (10<df['yloc'])
    df = df[~outsider_NE]
    df.loc[df['xloc'] < -10, 'xloc'] = 10 + (10 + df.loc[df['xloc'] < -10, 'xloc'])
    df.loc[10 < df['xloc'], 'xloc'] = -10 + (df.loc[10 < df['xloc'], 'xloc'] - 10)
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
    max_life = 1000000*.5
    nLife = len(df)
    cancer_realization = np.random.random(size = (nLife,1))
    df['tmp'] = cancer_realization
    cancer_result = (df['tmp'] < nLife/max_life)*(1-df['cancer_immunity'])
    df['will_die'] = (df['will_die']) | (cancer_result)
    return df



def get_nearest_lives(df):
    vicinity_distance = 0.0001
    nLife = len(df)
    life_dist = squareform(pdist(df[['xloc','yloc']]), checks = False)
    np.fill_diagonal(life_dist, np.inf)
    within_range = life_dist<[df['range_perception']]*nLife
    df['within_range'] = [np.where(x == True) for x in within_range]
    within_vicinity = life_dist < vicinity_distance*nLife
    df['within_vicinity'] = [np.where(x == True) for x in within_vicinity]
    df['nearest_life_loc'] = np.argmin(life_dist, axis = 1)
    return df

def chase_n_runaway(df):
    nLife = len(df)
    if nLife > 3:
        df = get_nearest_lives(df)
        df['tmp'] = df['within_range'].str[0].str[0]    # gives index of a life within range
        df['tmp2'] = np.random.random(size = (nLife,1))
        ischasing = (~df['tmp'].isna()) & (df['tmp2'] < df['chase_tendency'])
        ischased = list(df.loc[ischasing, 'tmp'].astype('int'))
        chase_xdir = df.iloc[ischased]['xloc'] - df.loc[ischasing, 'xloc']
        chase_ydir = df.iloc[ischased]['yloc'] - df.loc[ischasing, 'yloc']
        dist = np.sqrt(chase_xdir**2 + chase_ydir**2+1e-8)
        df.loc[ischasing, 'chase_xdir'] = chase_xdir/dist*3e-2
        df.loc[ischasing, 'chase_ydir'] = chase_ydir/dist*3e-2
        df.loc[~ischasing, 'chase_xdir'] = 0
        df.loc[~ischasing, 'chase_ydir'] = 0
        
        df['tmp2'] = np.random.random(size = (nLife,1))
        isrunningaway = (~df['tmp'].isna()) & (df['tmp2'] < df['cowardice'])
        isrunningawayfrom = list(df.loc[isrunningaway, 'tmp'].astype('int'))
        runaway_xdir = df.loc[isrunningaway, 'xloc'] - df.iloc[isrunningawayfrom]['xloc']
        runaway_ydir = df.loc[isrunningaway, 'yloc'] - df.iloc[isrunningawayfrom]['yloc']
        dist = np.sqrt(runaway_xdir**2 + runaway_ydir**2+1e-8)
        df.loc[isrunningaway, 'runaway_xdir'] = runaway_xdir/dist*3e-2
        df.loc[isrunningaway, 'runaway_ydir'] = runaway_ydir/dist*3e-2
        df.loc[~isrunningaway, 'runaway_xdir'] = 0
        df.loc[~isrunningaway, 'runaway_ydir'] = 0
    return df

def hunting(df):
    nLife = len(df)
    if nLife > 2:
        df = get_nearest_lives(df)
        df['tmp'] = df['within_vicinity'].str[0].str[0]    # gives index of a life within range
        df['tmp2'] = np.random.random(size = (nLife,1))
        ishunting = (~df['tmp'].isna()) & (df['tmp2'] < df['hunting_tendency'])
        ishunted = list(df.loc[ishunting, 'tmp'].astype('int'))
        df.loc[ishunting, 'energy'] += df.iloc[ishunted]['energy']*df.loc[ishunting, 'hunting_efficiency']
        df.loc[df['max_energy'] < df['energy'], 'energy'] = df.loc[df['max_energy'] < df['energy'], 'max_energy']
        df.iloc[ishunted]['will_die'] = True
    return df

def terminate_mortals(df):
    df = df[~df['will_die']]
    return df

