#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:53:04 2019

@author: joshs
"""
def get_lfp_channel_order():

        """
        Returns the channel ordering for LFP data extracted from NPX files.

        Parameters:
        ----------
        None

        Returns:
        ---------
        channel_order : numpy.ndarray
            Contains the actual channel ordering.
        """

        remapping_pattern = np.array([0, 12, 1, 13, 2, 14, 3, 15, 4, 16, 5, 17, 6, 18, 7, 19, 
              8, 20, 9, 21, 10, 22, 11, 23, 24, 36, 25, 37, 26, 38,
              27, 39, 28, 40, 29, 41, 30, 42, 31, 43, 32, 44, 33, 45, 34, 46, 35, 47])

        channel_order = np.concatenate([remapping_pattern + 48*i for i in range(0,8)])

        return channel_order

# %%

import numpy as np
import h5py as h5
import glob
from scipy.signal import butter, filtfilt, welch
from scipy.ndimage.filters import gaussian_filter1d
import os

mice = glob.glob('/mnt/md0/data/opt/production/*')

probes = ('probeA', 'probeB', 'probeC', 'probeD', 'probeE', 'probeF')

for folder in mice:
    
    mouse = folder[-6:]
    
    try:
        remote_server = create_samba_directory(experiment_dictionary[mouse][1][:3],experiment_dictionary[mouse][1])
    
        
        local_directory = '/mnt/md0/data/mouse' + mouse
    
        nwb_file = local_directory + '/mouse' + mouse + '.spikes.nwb'
        
        nwb = h5.File(nwb_file)
        
        for probe_idx, probe in enumerate(probes[:]):
        
            try:
                remote_directory = glob.glob(remote_server + '/*' + mouse + '*/*' + mouse + '*' + probe + '_sorted/continuous/Neuropix*100.1')[0]
        
                print(remote_directory)
        
                raw_data = np.memmap(remote_directory + '/continuous.dat', dtype='int16')
                data = np.reshape(raw_data, (int(raw_data.size / 384), 384))
                
                start_index = int(2500 * 1000) #(opto_times.value[trial_number]-0.5) - 3.759729003 * 30000)
                end_index = start_index+25000
               
                b,a = butter(3,[1/(2500/2),1000/(2500/2)],btype='band')
                
                order = get_lfp_channel_order()
                
                D = data[start_index:end_index,:]*0.195
            
                for i in range(D.shape[1]):
                   D[:,i] = filtfilt(b,a,D[:,order[i]])
                  
                M = np.median(D[:,370:])
                   
                for i in range(D.shape[1]):
                    D[:,i] = D[:,i] - M
                    
                channels = np.arange(D.shape[1])
                nfft = 2048
                    
                power = np.zeros((int(nfft/2+1), channels.size))
            
                for channel in range(D.shape[1]):
                    sample_frequencies, Pxx_den = welch(D[:,channel], fs=2500, nfft=nfft)
                    power[:,channel] = Pxx_den
                    
                in_range = (sample_frequencies > 0) * (sample_frequencies < 10)
        
                fig = plt.figure(frameon=False)
                plt.clf()
                fig.set_size_inches(1,8)
                
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                
                #plt.imshow(D.T, aspect='auto')
                S = np.std(D,0)
                S[S < 10] = np.nan
                S[S > 350] = np.nan
                ax.plot(np.mean(power[in_range,:],0),channels,'.',color='pink')
                #ax.plot(S,np.arange(384),alpha=0.5)
        
                unit_histogram = np.zeros((384,len(probes)),dtype='float')
                total_units = 0    
                     
                units = nwb['processing'][probe]['unit_list']
              
                modulation_index = np.zeros((len(units),))
                channels = np.zeros((len(units),))
                
                for unit_idx, unit in enumerate(units):
                    
                    channel = nwb['processing'][probe]['UnitTimes'][str(unit)]['channel'].value
                       
                    baseline = 1
                    evoked = 1
                    
                    #for t in trial_times[:100]:
                    #    baseline += np.sum((times > t - 0.25) * (times < t))    
                    #    evoked += np.sum((times > t) * (times < t + 0.25))  
                    
                    unit_histogram[channel,probe_idx] += 1 
                    
                    #if baseline > 50:
                    #modulation_index[unit_idx] = evoked/baseline #, channel, '.'
                    #channels[unit_idx] = channel
            
                    total_units += 1
                    
        
                GF = gaussian_filter1d(unit_histogram[:,probe_idx]*100,2.5)
                ax.barh(np.arange(384),GF,height=1.0,alpha=0.1,color='teal')
                ax.plot(GF,np.arange(384),linewidth=3.0,alpha=0.78,color='teal')    
                
                plt.ylim([0,384])
                plt.xlim([-5,400])
        
                #stop
                
                outpath = '/mnt/md0/data/opt/production/' + mouse + '/images'
                
                if not os.path.exists(outpath):
                    os.mkdir(outpath)
                
                fig.savefig(outpath + '/physiology_' + probe + '.png', dpi=300)   
                plt.close(fig)
                    
            except IndexError:
                pass
    except KeyError:
        pass
    
# %%
plt.figure(4517814)
plt.clf()

plt.plot(np.mean(power[in_range,:],0), np.arange(384),'.')


# %%    
surface_channels = [264, 216, 320, 260, 290]
scale_factor = 1.28

for i in range(0,5):
    
    #print('PROBE ' + probes[i])
    
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
    
    linepts = vv[0] * np.mgrid[-200:200:1][:,np.newaxis]
    linepts += datamean
    
    if linepts[-1,1] - linepts[0,1] < 0:
        linepts = np.flipud(linepts)
        
    intensity_values = np.zeros((linepts.shape[0],40))
    structure_ids = np.zeros((linepts.shape[0],))
    
    for j in range(linepts.shape[0]):
        
        z2,x2,y2 = transform.TransformFloatPoint(linepts[j,np.array([0,2,1])])
        
        ccf_coordinate = (np.array([1023-z2,x2,y2]) - origin) * scaling
        ccf_coordinate = ccf_coordinate[np.array([0,2,1])]
        
        ccf_coordinate_mm = ccf_coordinate * 0.01
        
        structure_ids[j] = int(labels[int(ccf_coordinate[0]),int(ccf_coordinate[1]),int(ccf_coordinate[2])]) - 1
        
        for k in range(-20,20):
            try:
                intensity_values[j,k+20] = volume[int(linepts[j,0]),int(linepts[j,1]+k),int(linepts[j,2]+k)]
            except IndexError:
                pass
            
    plt.subplot(1,6,i+1)
   
    borders = np.where(np.diff(structure_ids) > 0)[0]
    jumps = np.concatenate((np.array([5]),np.diff(borders)))
    borders = borders[jumps > 3] 
    border_loc = borders[0]*scale_factor - borders*scale_factor + surface_channels[i]
    
    #plt.plot(np.flip(structure_ids),np.arange(j+1))
    for i, border in enumerate(borders[::1]):
        if border_loc[i] < 0:
            plt.plot([0,400],[0, 0],'-',color='gray',alpha=0.5)
            plt.text(10,1,structure_tree[structure_tree.index == structure_ids[border]]['acronym'].iloc[0])
            break
        else: 
            plt.plot([0,400],[border_loc[i], border_loc[i]],'-',color='gray',alpha=0.5)
            plt.text(10,border_loc[i],structure_tree[structure_tree.index == structure_ids[border]]['acronym'].iloc[0])
        
        
    plt.xlim([0,400])
    #plt.text(0,0,structure_tree[structure_tree.index == structure_ids[-1]]['acronym'].iloc[0])
    #plt.ylim([0,j+1])
    #plt.axis('off')

# %%

raw_data = np.memmap(directory + '/continuous.dat', dtype='int16')
data = np.reshape(raw_data, (int(raw_data.size / 384), 384))


spike_times = np.load(directory + '/spike_times.npy')
spike_clusters = np.load(directory + '/spike_clusters.npy')
templates = np.load(directory + '/templates.npy')
whitening_mat_inv = np.load(directory + '/whitening_mat_inv.npy')
amplitudes = np.load(directory + '/amplitudes.npy')
channel_map = np.squeeze(np.load(directory + '/channel_map.npy'))

# %%



#trial_number = 5

start_index = int(30000 * 1000) #(opto_times.value[trial_number]-0.5) - 3.759729003 * 30000)
end_index = start_index+30000

channel_to_remove = np.array([36, 75, 112, 151, 188, 227, 264, 303, 340, 379])

plt.figure(148)
plt.clf()

b,a = butter(3,[300/(30000/2),2000/(30000/2)],btype='band')

D = data[start_index:end_index,channel_map]*0.195

plt.subplot(3,1,1)



plt.xlim([0,end_index-start_index])
plt.axis('off')

for i in range(0,D.shape[1],2):
    plt.plot(D[:,i]+i*15,'k',linewidth=0.1,alpha=0.25)
    

for i in range(D.shape[1]):
    D[:,i] = filtfilt(b,a,D[:,i])
    

    
D = pow(D / np.max(np.abs(D)),1)# * np.sign(D)

plt.subplot(3,1,2)
plt.imshow(D[:,1::2].T,
           vmin=-0.25,
           vmax=0.25,
           aspect='auto',
           origin='lower',
           cmap='RdGy')
plt.axis('off')

in_range = np.where((spike_times > start_index) * (spike_times < end_index))

times_in_range = spike_times[in_range]
ids_in_range = spike_clusters[in_range]
amps_in_range = amplitudes[in_range]

num_units = templates.shape[0]

Z = np.zeros(D.shape)

for idx, time in enumerate(times_in_range - start_index):

    if (time < Z.shape[0] - 42 and time > 40):
        template = templates[ids_in_range[idx],:,:]
    
        unwhitened_template = np.dot(np.ascontiguousarray(template),np.ascontiguousarray(whitening_mat_inv))
        
        Z[int(time-40):int(time-40+82),:] += unwhitened_template * amps_in_range[idx]
    
plt.subplot(3,1,3)
plt.imshow(Z[:,1::2].T,
           vmin=-400,
           vmax=400,
           aspect='auto',
           origin='lower',
           cmap='RdGy')
plt.axis('off')



#plt.axis('off')
    