import numpy as np
from numpy import genfromtxt
from custom_function import *
from PIL import Image

import sys
import os

defect_par_loc = #'/data01/user-storage/zitong/data/random_anatomy_3D/phantom/castor/'
healthy_par_loc = #'/data02/user-storage/zitong/data/random_anatomy_3D/phantom/castor/'

defect_rec_loc = #'/data01/user-storage/zitong/data/random_anatomy_3D/phantom/castor/test_npy/'
healthy_rec_loc = #'/data02/user-storage/zitong/data/random_anatomy_3D/phantom/castor/test_npy/'

defect_folder_list = #['1_0.5','2_0.25','4_0.5','5_0.25'] # four different defect types 
# dose_lvl_list = [1,0.2,0.15,0.1,0.05] # dose level

print('loading images...')
normal_g = {'d':{},'h':{}} # normal_g[ defect or healthy ][ sub-ensumble ]

subensemble_idx = 0
for defect_type in defect_folder_list:
    for dose_lvl in [1]:
        for patient_idx in np.arange(101,201):

            defect_par = f'{defect_par_loc}/{defect_type}/{patient_idx}/parfiles/normal.par'
            current_defect_rec = np.load(f'{defect_rec_loc}{dose_lvl}/{defect_type}/{patient_idx}.npy')

            healthy_par = f'{healthy_par_loc}/{defect_type}/{patient_idx}/parfiles/normal.par'
            current_healthy_rec = np.load(f'{healthy_rec_loc}{dose_lvl}/{defect_type}/{patient_idx}.npy')

            # d_x, d_y, d_z: Coordinates of defect center location
            if defect_type in ['1_0.5','2_0.25']:
                d_x = 71
                if d_gender == 1:
                    d_y = 57
                else:
                    d_y = 59
                d_z = 34
            else:
                d_x = 71
                if d_gender == 1:
                    d_y = 61
                else:
                    d_y = 63
                d_z = 43

            current_defect_extracted = current_defect_rec[d_x-16:d_x+16,d_y-16:d_y+16,d_z-16:d_z+16]
            current_defect_extracted = current_defect_extracted.flatten()
            current_defect_extracted = (current_defect_extracted-np.min(current_defect_extracted))/(np.max(current_defect_extracted)-np.min(current_defect_extracted))*255
            current_healthy_extracted = current_healthy_rec[d_x-16:d_x+16,d_y-16:d_y+16,d_z-16:d_z+16]
            current_healthy_extracted = current_healthy_extracted.flatten()
            current_healthy_extracted = (current_healthy_extracted-np.min(current_healthy_extracted))/(np.max(current_healthy_extracted)-np.min(current_healthy_extracted))*255
            try:
                normal_g['d'][subensemble_idx].append(current_defect_extracted)
                normal_g['h'][subensemble_idx].append(current_healthy_extracted)
            except:
                normal_g['d'][subensemble_idx] = []
                normal_g['d'][subensemble_idx].append(current_defect_extracted)
                normal_g['h'][subensemble_idx] = []
                normal_g['h'][subensemble_idx].append(current_healthy_extracted)

    subensemble_idx = subensemble_idx + 1

print('loading pre-written 3D channels file...')
U = np.load(f'U3d_a7.npy')
print('U shape: ',U.shape,'U a: ',a)
U = np.transpose(U,[1,0])

###################################################################

print('calculating test statistics...')
tS_normal = []
tN_normal = []

delta_g_bar_normal = np.zeros((1,32*32*32))
for i in np.arange(0,4):
    print('sub-ensemble: '+str(i))
    tS, tN = MultiLDpooled(normal_g['d'][i],normal_g['h'][i],U)
    tS_normal.append(tS)
    tN_normal.append(tN)

tS_normal = np.concatenate(tS_normal)
tN_normal = np.concatenate(tN_normal)
np.save(f'./tS_1.npy',tS_normal)
np.save(f'./tN_1.npy',tN_normal)

print('done')
