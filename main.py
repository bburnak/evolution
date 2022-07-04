# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 20:20:26 2022

@author: baris
"""
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib.patches import Polygon
from matplotlib.path import Path
import numpy as np
import pandas as pd
from functions import *
from functions_map import *
fig, ax = plt.subplots(figsize = (15,15))
scale = 5
nSources = 8
x, y = [np.random.rand(1)*scale],[np.random.rand(1)*scale]
color = [np.random.rand(1)]
transparency = [np.ones(1)]
prob_reproduce = 0.03
prob_colorChange = [np.random.rand(1)*1e-2]
prob_rest = [0.8]
energy = [40]
athleticism = [0.1]
graze_efficiency = [0.1]
base_metabolism = [0.1]

source_polygons = []
for i in range(nSources):
    source_coordinates = generate_valley()
    source_polygon = Path(source_coordinates)
    source_polygons.append(source_polygon)
    energy_source_mapped = Polygon(source_coordinates, color='g', alpha = 0.15)
    ax.add_patch(energy_source_mapped)

sc = ax.scatter(x,y, c = color, cmap = 'copper', vmin = 0, vmax = 1)  
ax.set_xticks([])
ax.set_yticks([])
plt.xlim(-10,10)
plt.ylim(-10,10)

d = {'xloc': x, 'yloc': y, 'colors': color, 'transparency': transparency,
     'prob_reproduce': prob_reproduce,
     'prob_colorChage': prob_colorChange,
     'prob_rest': prob_rest,
     'athleticism': athleticism,
     'will_die': False,
     'energy': energy,
     'graze_efficiency': graze_efficiency,
     'max_energy': energy,
     'mutation': 0,
     'base_metabolism': base_metabolism}

df = pd.DataFrame(data = d)

def animate(i):
    global df
    df = calc_loc(df)
    df = calc_reproduction(df)
    df = calc_colors(df)
    df = kill_outsiders(df)
    df = grazing(df)
    df = get_cancer(df)
    df = terminate_mortals(df)
    
    sc.set_offsets(np.c_[df['xloc'],df['yloc']])
    sc.set_array(df['colors'].astype('float'))
    sc.set_alpha(df['energy_ratio'].astype('float'))
    ax.set_title('Number of life: ' + str(len(df)))
ani = matplotlib.animation.FuncAnimation(fig, animate, 
                                         frames = 2, interval = 2, repeat = True)
plt.show()

def grazing(df):
    global source_polygons
    xy = df[['xloc','yloc']]
    df['grazing'] = False
    for polygon in source_polygons:
        df['grazing'] = df['grazing'] | polygon.contains_points(xy)
    df['energy_gain'] = df['graze_efficiency']
    df.loc[df['grazing'],'energy'] += df.loc[df['grazing'],'energy_gain']
    df.loc[df['max_energy'] < df['energy'], 'energy'] = df.loc[df['max_energy'] < df['energy'], 'max_energy']
    return df