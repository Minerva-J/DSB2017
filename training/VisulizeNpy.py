# -- coding:utf-8 --
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

data_dir = '/home/zhaojie/zhaojie/Lung/DSB_Code/DSB2017-master/training/Data/ForTest/Preprocess/'
filenames = os.path.join(data_dir, 'T001351128_clean.npy')
imgs = np.load(filenames)[0]
print('------0',imgs.shape,np.max(imgs),np.min(imgs))
from scipy import misc
for i in range(imgs.shape[0]):
    name = str(i) + '_' + '.png'
    cv2.imwrite(os.path.join('./CT_cleannp/', name), imgs[i])