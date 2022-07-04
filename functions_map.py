import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

def generate_valley():
    scale = np.random.uniform(0,8)
    points = np.random.rand(60,2)*scale
    hull = ConvexHull(points)
    x,y = points[hull.vertices,0], points[hull.vertices,1]
    x += np.random.uniform(-10,10)
    y += np.random.uniform(-10,10)
    xy = np.vstack([x,y]).T
    return xy