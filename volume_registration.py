#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 15:35:01 2019

@author: joshs
"""


import vtk

import numpy as np
import pandas as pd

def loadVolume(fname, _dtype='u1', num_slices=1023):
        
    dtype = np.dtype(_dtype)

    volume = np.fromfile(fname, dtype) # read it in
    
    print(volume[:13])
    
    z_size = np.sum([volume[1], volume[2] << pow(2,3)])
    x_size = np.sum([(val << pow(2,i+1)) for i, val in enumerate(volume[8:4:-1])])
    y_size = np.sum([(val << pow(2,i+1)) for i, val in enumerate(volume[12:8:-1])])
    
    fsize = np.array([z_size, x_size, y_size]).astype('int')

    print(fsize)

    volume = np.reshape(volume[13:], fsize) # remove 13-byte header and reshape
    
    return volume

# %%

mouse = '438844'

base_dir = r"\\10.128.50.77\sd5.2\OPT\processed"
mouse_dir = os_join(base_dir, mouse)
template_dir = os_join(base_dir, 'template_brain')

fname = os_join(mouse_dir, '/probe_annotations.npy')
probe_annotations = np.load(fname)

volume = loadVolume(os_join(mouse_dir, '_trans.pvl.nc.001'))

template_dir = os_join(base_dir, 'template_brain')
template = loadVolume(os_join(template_dir, 'template_fluor.pvl.nc.001'))
labels = np.load(os_join(template_dir, 'area_labels.npy'))

source_landmarks = np.load(os_join(mouse_dir, '/landmark_annotations.npy'))
target_landmarks = np.load(os_join(template_dir, 'landmark_annotations.npy'))

source_landmarks = source_landmarks[:,np.array([2,0,1])]
target_landmarks = target_landmarks[:,np.array([2,0,1])]

structure_tree = pd.read_csv(os_join(template_dir, 'ccf_structure_tree.csv'))

# %%


# STEP 1: define transform

transform = vtk.vtkThinPlateSplineTransform()

source_points = vtk.vtkPoints()
target_points = vtk.vtkPoints()

for x in [0,1024]:
    for y in [0,1024]:
        for z in [0,1023]:
            source_points.InsertNextPoint([z,x,y])
            target_points.InsertNextPoint([z,x,y])

for i in range(source_landmarks.shape[0]):
    if source_landmarks[i,0] > -1 and target_landmarks[i,0] > -1:
        source_points.InsertNextPoint(source_landmarks[i,:])
    
for i in range(target_landmarks.shape[0]):
    if source_landmarks[i,0] > -1 and target_landmarks[i,0] > -1:
        target_points.InsertNextPoint(target_landmarks[i,:])

transform.SetBasisToR() # for 3D transform
transform.SetSourceLandmarks(source_points)
transform.SetTargetLandmarks(target_points)
transform.Update()


colors = ('red', 'orange', 'brown', 'green', 'blue', 'purple')

probes = ('probeA', 'probeB', 'probeC', 'probeD', 'probeE', 'probeF')


plt.figure(147142)
plt.clf()

plt.figure(147143)
plt.clf()


output_figures = False


origin = np.array([-35, 42, 217],dtype='int')
scaling = np.array([1160/1023,  1140/940, 800/590])

df_columns = ['probe','structure_id', 'A/P','D/V','M/L']

df = pd.DataFrame(columns = df_columns)

for i in range(0,6):
    
    print('PROBE ' + probes[i])
    
    x = probe_annotations[:,0,i]
    y = probe_annotations[:,1,i]
    
    z = np.where(x > -1)[0]
    x = x[x > -1]
    y = y[y > -1]
    
    if len(z) > 0:
        
        data = np.vstack((z,y,x)).T
        datamean = data.mean(axis=0)
        D = data - datamean
        m1 = np.min(D[:,1]) * 2
        m2 = np.max(D[:,1]) * 2
        uu,dd,vv = np.linalg.svd(D)
        
        linepts = vv[0] * np.mgrid[-200:200:0.7][:,np.newaxis]
        linepts += datamean
        
        if linepts[-1,1] - linepts[0,1] < 0:
            linepts = np.flipud(linepts)
            
        plt.figure(147142)
        plt.subplot(1,2,1)
        plt.plot(linepts[:,0],linepts[:,1],color=colors[i])
        plt.plot(z,y,'.',color=colors[i])
        plt.subplot(1,2,2)
        plt.plot(linepts[:,1],linepts[:,2],color=colors[i])
        plt.plot(y,x,'.',color=colors[i])
        
        intensity_values = np.zeros((linepts.shape[0],40))
        structure_ids = np.zeros((linepts.shape[0],))
        ccf_coordinates = np.zeros((linepts.shape[0],3))
        
        for j in range(linepts.shape[0]):
            
            z2,x2,y2 = transform.TransformFloatPoint(linepts[j,np.array([0,2,1])])
            
            ccf_coordinate = (np.array([1023-z2,x2,y2]) - origin) * scaling
            ccf_coordinate = ccf_coordinate[np.array([0,2,1])]
            
            ccf_coordinate_mm = ccf_coordinate * 0.01
            
            ccf_coordinates[j,:] = ccf_coordinate_mm
            
            structure_ids[j] = int(labels[int(ccf_coordinate[0]),int(ccf_coordinate[1]),int(ccf_coordinate[2])]) - 1
            
            for k in range(-20,20):
                try:
                    intensity_values[j,k+20] = volume[int(linepts[j,0]),int(linepts[j,1]+k),int(linepts[j,2]+k)]
                except IndexError:
                    pass
                
        data = {'probe': [probes[i]]*linepts.shape[0], 
                'structure_id': structure_ids.astype('int'), 
                'A/P' : np.around(ccf_coordinates[:,0],3), 
                'D/V' : np.around(ccf_coordinates[:,1],3), 
                'M/L' : np.around(ccf_coordinates[:,2],3) 
                }

        probe_df = pd.DataFrame(data)
    
        df = pd.concat((df, probe_df) ,ignore_index=True)
        
        plt.figure(147143)
        plt.subplot(1,12,i*2+1)
        plt.imshow(intensity_values, cmap='gray',aspect='auto')
        plt.plot([20,20],[0,j],'-r')
        plt.axis('off')
        
        if output_figures:
            fig = plt.figure(frameon=False)
            fig.set_size_inches(1,8)
            
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            
            ax.imshow(intensity_values, cmap='gray',aspect='auto')
            fig.savefig('/mnt/md0/data/opt/production/' + mouse + '/images/histology_' + probes[i] + '.png', dpi=300)    
            
            plt.close(fig)
                
        borders = np.where(np.diff(structure_ids) > 0)[0]
        jumps = np.concatenate((np.array([5]),np.diff(borders)))
        borders = borders[jumps > 3]
        
        for border in borders[::1]:
            plt.plot([0,40],[border,border],'-',color='white',alpha=0.5)
            
        plt.subplot(1,12,i*2+2)
        for border in borders[::1]:
            plt.text(0,j-border-1,structure_tree[structure_tree.index == structure_ids[border-1]]['acronym'].iloc[0])
            
        plt.text(0,0,structure_tree[structure_tree.index == structure_ids[-1]]['acronym'].iloc[0])
        plt.ylim([0,j+1])
        plt.axis('off')
        
df.to_csv(output_file)

    # %%

    
# %%
    
plt.figure(4171)
plt.clf()


struct_id = 669

a = np.zeros(labels.shape)
a[labels == struct_id] = 1

plt.subplot(1,3,1)
plt.imshow(np.max(a,0))

plt.subplot(1,3,2)
plt.imshow(np.max(a,1))

plt.subplot(1,3,3)
plt.imshow(np.max(a,2))    
        

# %%
        
        
        
        if struct_id > 0:
    
            struct_name = structure_tree[structure_tree.index == struct_id]['acronym'].iloc[0]
            print(struct_name)

# %%\
# splines:
from scipy.interpolate import splprep, splev

tck, u = splprep([D[:,0],D[:,1],D[:,2]], s=1000, t=1,k=1)
new_points = splev(np.linspace(-0.1,1.1,20), tck, der=0)

# %%%

origin = np.array([-35, 42, 217],dtype='int')
scaling = np.array([1160/1023,  1140/940, 800/590])


for i in range(0,6):
    
    print('PROBE ' + probes[i])
    
    x = probe_annotations[:,0,i]
    y = probe_annotations[:,1,i]
    
    z = np.where(x > -1)[0]
    x = x[x > -1]
    y = y[y > -1]
    
    data = np.vstack((z,y,x)).T
    datamean = data.mean(axis=0)
    D = data - datamean
    m1 = np.min(D[:,1]) * 2
    m2 = np.max(D[:,1]) * 2
    uu,dd,vv = np.linalg.svd(D)
    
    linepts = vv[0] * np.mgrid[-250:250:10][:,np.newaxis]
    linepts += datamean
    
    if linepts[-1,2] - linepts[0,2] > 0:
        linepts = np.flipud(linepts)
    
    for j in range(linepts.shape[0]):
        z2,x2,y2 = transform.TransformFloatPoint(linepts[j,np.array([0,2,1])])
        
        struct_id = int(labels[int(z2),int(y2),int(x2)])
        
        ccf_coordinate = (np.array([1023-z2,x2,y2]) - origin) * scaling
        ccf_coordinate = ccf_coordinate[np.array([0,2,1])]
        
        ccf_coordinate_mm = ccf_coordinate * 0.01
        
        if struct_id > 0:

            struct_name = structure_tree[structure_tree.index == struct_id]['acronym'].iloc[0]
            print(struct_name)
            
    print( ' ' )
            
        #if j == 20:
        #    plt.figure(4812341)
        #    plt.clf()
            
        #    plt.subplot(1,2,1)
        #    plt.imshow(template[int(z2),:,:])
        #    plt.plot(x2,y2,'.',color='white')
            
        #    plt.subplot(1,2,2)
        #    plt.imshow(original_template[int(ccf_coordinate[0]),:,:])
        #    plt.plot(ccf_coordinate[2],ccf_coordinate[1],'.',color='white')
        #    plt.title(struct_name)
            
        #    stop

# %%

# STEP 2: create new volume
    
from scipy.ndimage import sobel, gaussian_filter
    
plt.figure(142741)
plt.clf()

test_slices = range(200,600,50)
    
for slice_i, test_slice in enumerate(test_slices): 

    vol2 = np.zeros(volume.shape)
    
    for zi in range(test_slice,test_slice+3): #vol2.shape[2]):
    
        print("Slice " + str(zi))
        
        for xi in range(vol2.shape[1]):
            for yi in range(vol2.shape[2]):
        
                z,x,y = transform.TransformFloatPoint([zi,xi,yi])
                
                if x < vol2.shape[1] and x > 0 and \
                   y < vol2.shape[2] and y > 0 and \
                   z < vol2.shape[0] and z > 0:
                       vol2[int(z),int(y),int(x)] = volume[zi,yi,xi]
    
    slice_num = int(z)

    im = labels[slice_num,:,:]
    grad1 = np.gradient(im)[0] / 32767 * 200
    grad2 = np.flipud(np.gradient(np.flipud(im))[0] / 32767 * 200)
    grad = gaussian_filter(grad1 + grad2,1)       

    plt.subplot(2,4,slice_i+1)
    plt.imshow(np.max(vol2[:,:,:]+grad,0), cmap='gray')    
    plt.axis('off')        
                   
            
# %%
                   




slice_num = int(z)

im = labels[slice_num,:,:]
grad1 = np.gradient(im)[0] / 32767 * 200
grad2 = np.flipud(np.gradient(np.flipud(im))[0] / 32767 * 200)
grad = gaussian_filter(grad1 + grad2,1)

plt.subplot(1,3,1)
plt.imshow(np.max(vol2[:,:,:]+grad,0), cmap='gray')

plt.subplot(1,3,2)
plt.imshow(template[slice_num,:,:]+grad, cmap='gray')

plt.subplot(1,3,3)
plt.imshow(grad, cmap='gray')
