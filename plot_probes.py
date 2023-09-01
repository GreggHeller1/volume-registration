#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:30:08 2019

@author: joshs
"""


mouse = '421529'
    
fname = '/mnt/md0/data/opt/production/' + mouse + '/mouse' + mouse + '_trans.pvl.nc.001'

volume1 = loadVolume(fname)

fname = '/mnt/md0/data/opt/production/' + mouse + '/mouse' + mouse + '_fluor.pvl.nc.001'

volume2 = loadVolume(fname)

# %%

fname = '/mnt/md0/data/opt/production/' + mouse  + '/probe_annotations.npy'

d = np.load(fname)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(999)
plt.clf()

ax = fig.add_subplot(111, projection='3d')

fig = plt.figure(1000)
plt.clf()

probe_names = ['A', 'B','C','D','E','F']

for i in range(0,6):
    
    x = d[:,0,i]
    y = d[:,1,i]
    
    z = np.where(x > -1)[0]
    x = x[x > -1]
    y = y[y > -1]
    
    data = np.vstack((z,y,x)).T
    datamean = data.mean(axis=0)
    D = data - datamean
    m1 = np.min(D[:,1]) * 2
    m2 = np.max(D[:,1]) * 2
    uu,dd,vv = np.linalg.svd(D)
    
    linepts = vv[0] * np.mgrid[-250:250:2][:,np.newaxis]
    linepts += datamean
    
    plt.figure(999)
    
    ax.scatter(z,y,x)
    ax.plot3D(*linepts.T)
    plt.axis('square')
    plt.axis('equal')


    plt.figure(1000)
    
    density1 = np.zeros((linepts.shape[0],50))
    density2 = np.zeros(density1.shape)

    for j in range(linepts.shape[0]):
        for t in range(-25,25):
            density1[j,t+25] = volume1[int(linepts[j,0])+t,int(linepts[j,1]),int(linepts[j,2])]
            density2[j,t+25] = volume2[int(linepts[j,0])+t,int(linepts[j,1]),int(linepts[j,2])]
    
    density = np.zeros((linepts.shape[0],50,3))
    density[:,:,0] = density2 / 255
    density[:,:,1] = density1 / 400
    density[:,:,2] = density1 / 400
    
    if linepts[0,1] > linepts[-1,1]:
        density = np.flipud(density)
    
    plt.subplot(1,6,i+1)
    plt.imshow(density) #, cmap='gray',aspect='auto',vmin=0,vmax=255)
    
    plt.title(probe_names[i])
    plt.axis('off')
    
# %%

plt.figure(1814)
plt.clf()

plt.imshow(volume[:,:,131])
    
    
# %%


# %%

for i in linepts.shape[0]:
    
    

# %%

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
