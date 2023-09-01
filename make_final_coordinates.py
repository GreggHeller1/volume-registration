#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:36:01 2019

@author: joshs
"""

# %%

import glob, os, re

mouse = '406807'

fname = '/mnt/md0/data/opt/production/' + mouse + '/final_ccf_coordinates.csv'

cortex_labels = pd.read_csv('/mnt/md0/data/production_QC/isi_overlay/cortex_ccf_labels.csv',index_col=0)

df = pd.read_csv(fname)
structure_tree = pd.read_csv('/mnt/md0/data/opt/template_brain/ccf_structure_tree_2017.csv')

# %%

probes = ('probeA', 'probeB', 'probeC', 'probeD', 'probeE', 'probeF')

for probe in probes:
    
    coords = np.zeros((384,3))
         
    structures = ['none'] * 384
    
    sub_table = df[df.probe == probe]
    channels = sub_table.channels.values
    
    for channel in range(384):
        
        index = np.argmin(np.abs(channels - channel))
        
        structure_id = sub_table.iloc[index]['structure_id']
         
        name = structure_tree[structure_tree.index == structure_id]['acronym'].iloc[0]
        name = name.split('-')[0]
        
        numbers = re.findall(r'\d+', name)
            
        if len(numbers) > 0 and name[:2] != 'CA':
            name = cortex_labels.loc[int(mouse)][probe] + '/'.join(numbers)
         
        if name == 'root':
            name = 'none'
            
        structures[channel] = name
        
        last_good = 'none'
        found_surface = False
        
        for idx, acronym in enumerate(structures):
            
            if acronym.islower() and not found_surface:
                structures[idx] = last_good
            else:
                last_good = acronym
                if acronym[-1] == '1' and acronym[:2] != 'CA':
                    found_surface = True
        
        coords[channel,:] = np.array([ sub_table.iloc[index]['A/P'], 
                                       sub_table.iloc[index]['D/V'], 
                                       sub_table.iloc[index]['M/L'], 
                                     ])
        
    data = {'acronym': structures, 
            'A/P' : np.around(coords[:,0],3), 
            'D/V' : np.around(coords[:,1],3), 
            'M/L' : np.around(coords[:,2],3) 
            }
    
    directory = glob.glob('/mnt/md0/data/mouse' + mouse + '/*' + probe + '_sorted/continuous/Neuropix*.0')[0]
    fname = os.path.join(directory, 'ccf_regions_new.csv')
     
    df2 = pd.DataFrame(data = data)
    df2.to_csv(fname)
         
         