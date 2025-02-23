import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from multiprocessing import Pool
import multiprocessing   
import os
import sys

#We add points so the boundary conditions are satisfied
#Because the Voronoi teselation would give incorrect (infinite) regions for the border points, we 'apply' the periodic conditions 
# by copying for each boundary a range of points of the opposite boundary and translating them a length equal to the box length (29)
def extended_points(df1):
    df1_copy2=df1.copy()
    df1_copy=pd.concat([df1_copy2,df1_copy2.loc[df1_copy2['x'] > 26, ['label','type','x','y']] - [0,0,29,0],
                        df1_copy2.loc[df1_copy2['x'] < 2, ['label','type','x','y']] + [0,0,29,0]],
                    ignore_index=True)
    df1_copy=pd.concat([df1_copy,
                        df1_copy.loc[df1_copy['y'] > 26, ['label','type','x','y']] - [0,0,0,29],        
                        df1_copy.loc[df1_copy['y'] < 2, ['label','type','x','y']] + [0,0,0,29]],
                    ignore_index=True)
    return df1_copy

#Functions for diferent Voronoi features
def polygon_area(vertices):
    n=len(vertices)
    area=0
    for i in range(n):
        j=(i+1)%n
        area+=vertices[i][0]*vertices[j][1]
        area-=vertices[j][0]*vertices[i][1]
    area=abs(area)/2
    return area

def polygon_perimeter(vertices):
    n=len(vertices)
    perimeter=0
    for i in range(n):
        j=(i+1)%n
        perimeter+=np.sqrt((vertices[i][0]-vertices[j][0])**2+(vertices[i][1]-vertices[j][1])**2)
    return perimeter

def centroid(vertices,area):
    n=len(vertices)
    c=np.array([0.0,0.0])
    for i in range(n):
        j=(i+1)%n
        c+=(vertices[i][0]*vertices[j][1]-vertices[j][0]*vertices[i][1])*np.array([vertices[i][0]+vertices[j][0],vertices[i][1]+vertices[j][1]])
    return abs(c[0]/(6*area)),abs(c[1]/(6*area))

def voronoi_max_min_distances(points,vor):
    n=len(points)
    neighbours={i:set() for i in range(n)}

    for pair in vor.ridge_points:
        neighbours[pair[0]].add(pair[1])
        neighbours[pair[1]].add(pair[0])
    max_list=np.zeros(n)
    min_list=np.zeros(n)
    for i in range(n):
        if neighbours[i]:
            dist=[np.linalg.norm(points[i]-points[j]) for j in neighbours[i]]
            max_list[i]=max(dist)
            min_list[i]=min(dist)
    return max_list,min_list

def max_min_vertices_length(points,vor):
    n=len(points)
    vertices_coor=vor.vertices
    regions=vor.point_region
    max_length = np.full(n, np.nan)
    min_length = np.full(n, np.nan)
    
    for i,region_idx in enumerate(regions):
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            continue
            
        dist=[]
        for k in region:
            d=[np.linalg.norm(vertices_coor[k]-vertices_coor[j]) for j in region if j>k]
            dist=dist+d
        if dist:
            max_length[i]=max(dist)
            min_length[i]=min(dist)    
    return max_length,min_length

def max_min_vertices_point_length(points,vor):
    n=len(points)
    vertices_coor=vor.vertices
    regions=vor.point_region
    max_length = np.full(n, np.nan)
    min_length = np.full(n, np.nan)
    
    for i,region_idx in enumerate(regions):
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            continue
            
        dist=[np.linalg.norm(points[i]-vertices_coor[j]) for j in region]
        max_length[i]=max(dist)
        min_length[i]=min(dist)    
    return max_length,min_length

#Function that obtains the voronoi features for a set of points and returns a dataframe with them
def voronoi_parameters(points):
    vor = Voronoi(points)
    areas = np.zeros(len(points))
    perimeters = np.zeros(len(points))
    neighbours= np.zeros(len(points))
    centersx= np.zeros(len(points))
    centersy= np.zeros(len(points))

    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:  
            areas[i] = np.nan  
            perimeters[i]=np.nan
            neighbours[i]=np.nan
            centersx[i]=np.nan
            centersy[i]=np.nan
        else:
            polygon = np.array([vor.vertices[j] for j in region])
            areas[i] = polygon_area(polygon)
            perimeters[i]=polygon_perimeter(polygon)
            neighbours[i]=len(polygon)
            centersx[i],centersy[i]=centroid(polygon,areas[i])
    max_neighbours,min_neighbours=voronoi_max_min_distances(points,vor)
    max_vertices,min_vertices=max_min_vertices_length(points,vor)
    max_vertices_point,min_vertices_point=max_min_vertices_point_length(points,vor)
    
    return areas,perimeters,neighbours,centersx,centersy,max_neighbours,min_neighbours,max_vertices,min_vertices,max_vertices_point,min_vertices_point

def dead_alive(type):
    if type>2:
        return 0
    else:
        return 1
    
def particle_type(type):
    if (type==1 or type==3):
        return 1
    else: 
        return 0

#Function that delete the extra points and add two columns: activity and particle type
def features(df):
    df11=(extended_points(df)).copy()
    points=np.array(extended_points(df)[['x','y']])
    df11['area'],df11['perimeter'],df11['neighbours'],df11['centerx'],df11['centery'],df11['max neighbour distance'],df11['min neighbour distance'],df11['max vertices distance'],df11['min vertices distance'],df11['max vertices-point distance'],df11['min vertices-point distance']=voronoi_parameters(points)
    df11['distance to center']=np.sqrt((df11['x']-df11['centerx'])**2+(df11['y']-df11['centery'])**2)
    df12=df11.drop(df11[(df11['x']>29) | (df11['x']<0) | (df11['y']>29) | (df11['y']<0)].index)
    df12['activity']=df12['type'].apply(dead_alive)
    df12['particle type']=df12['type'].apply(particle_type)
    df12=df12.drop(columns=['x','y','label','type','centerx','centery'])
    #And we recover a 1000 rows df  
    return df12

# Function that process an image (n_columns particles) and extracts its features
def process_image(df,n_columns): 
    df.columns = ["label", "type", "x", "y"]
    df1=features(df)
    data=df1.to_numpy().reshape(-1,n_columns)
    return data

density=sys.argv[1]
fa=sys.argv[2]
input_file=f'phia{density}/traj_phia{density}-T05-Fa{fa}-tau1.dat'
output_file=f"phia{density}/particles-features-{density}-Fa{fa}.txt"

# File parameters
def n_rows(input_file):
    df=pd.read_csv(input_file, sep='\s+')
    return df.shape[0]
num_rows = n_rows(input_file)  # Total number of rows
rows_per_image = 1000    
num_images = num_rows // rows_per_image

#Adjust the stride to your PC specs by watching the % of CPU that is used while executing
#If you select a stride of 1 (not using multiprocessing) the code should execute in around 15 minutes, with stride=10 it executed in less than 3 in my PC
stride=10

#Images are being saved in a disordered way in the map results due to multiprocessing, 
#that`s why the processing image is shown in a random way, 
#later, before we write the images on the output file, they are sorted by their index so no problem
    
def save_features(input):
    if input[0]%10==0: 
        print(f"Processing image {input[0]+1}/{num_images}...")
    # Read only the necessary columns for processing image

    df = pd.read_csv(input[1], sep='\s+', header=None, skiprows=input[0]*rows_per_image, nrows=rows_per_image)
    data_with_features = (input[0],process_image(df,12))
    # Save in file
    return data_with_features


if __name__=='__main__':  
    with Pool(processes=stride) as p:
        results=p.map(save_features,[(i,input_file) for i in range(num_images)])
    with open(output_file, "w") as f_out:
        results.sort(key=lambda e:e[0])
        for result in results:
            np.savetxt(f_out, result[1], fmt="%.6f")    