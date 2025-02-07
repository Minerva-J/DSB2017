import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
import collections
import random
from layers import iou
from scipy.ndimage import zoom
import warnings
from scipy.ndimage.interpolation import rotate
from layers import nms,iou
import pandas

class DataBowl3Classifier(Dataset):
    def __init__(self, split, config, phase = 'train'):
        assert(phase == 'train' or phase == 'val' or phase == 'test')
        
        self.random_sample = config['random_sample']#True
        self.T = config['T']#1
        self.topk = config['topk']#5
        self.crop_size = config['crop_size']#[96,96,96]
        self.stride = config['stride']#4
        self.augtype  = config['augtype']#{'flip':True,'swap':False,'rotate':False,'scale':False}
        #self.labels = np.array(pandas.read_csv(config['labelfile']))
        
        datadir = config['datadir']#preprocess_result_path
        bboxpath  = config['bboxpath']#'../detector/results/res18/bbox/',
        self.phase = phase
        self.candidate_box = []
        self.pbb_label = []
        
        split = [str(f,'utf-8') for f in split]
        idcs = split
        self.filenames = [os.path.join(datadir, '%s_clean.npy' % idx) for idx in idcs]
        labels = np.array(pandas.read_csv(config['labelfile']))#'./full_label.csv'
        if phase !='test':
            # print('0',labels[:,0])
            # print('0=', [str(f,'utf-8') for f in split])
            # self.yset = np.array([labels[labels[:,0]==str(f,'utf-8'),1] for f in split]).astype('int')
            self.yset = np.array([labels[labels[:,0]==f.split('-')[0].split('_')[0],1] for f in split]).astype('int')
        # print([str(f,'utf-8') for f in idcs])
        # idcs = [str(f,'utf-8') for f in idcs]
        
        
        for idx in idcs:
            pbb = np.load(os.path.join(bboxpath,idx+'_pbb.npy'))
            # print('pbb.shape',pbb.shape)#(~,5)
            pbb = pbb[pbb[:,0]>config['conf_th']]#-1
            pbb = nms(pbb, config['nms_th'])#0.05
            
            lbb = np.load(os.path.join(bboxpath,idx+'_lbb.npy'))
            pbb_label = []
            
            for p in pbb:
                isnod = False
                for l in lbb:
                    score = iou(p[1:5], l)
                    if score > config['detect_th']:#0.05
                        isnod = True
                        break
                pbb_label.append(isnod)
#             if idx.startswith()
            self.candidate_box.append(pbb)
            self.pbb_label.append(np.array(pbb_label))
        self.crop = simpleCrop(config,phase)
        

    def __getitem__(self, idx,split=None):
        t = time.time()
        np.random.seed(int(str(t%1)[2:7]))#seed according to time
        # print('idx',idx)
        pbb = self.candidate_box[idx]
        pbb_label = self.pbb_label[idx]
        conf_list = pbb[:,0]
        T = self.T
        topk = self.topk
        img = np.load(self.filenames[idx])
        # print('-',img.shape,self.filenames[idx])#(1, 320, 206, 322) ../Data/preprocess/d92998a73d4654a442e6d6ba15bbb827_clean.npy
        if self.random_sample and self.phase=='train':
            chosenid = sample(conf_list,topk,T=T)
            #chosenid = conf_list.argsort()[::-1][:topk]
        else:
            chosenid = conf_list.argsort()[::-1][:topk]
        croplist = np.zeros([topk,1,self.crop_size[0],self.crop_size[1],self.crop_size[2]]).astype('float32')
        # print(self.crop_size[0], self.stride, self.crop_size[1], self.stride, self.crop_size[2], self.stride)
        coordlist = np.zeros([topk,3,self.crop_size[0]//self.stride,self.crop_size[1]//self.stride,self.crop_size[2]//self.stride]).astype('float32')
        padmask = np.concatenate([np.ones(len(chosenid)),np.zeros(self.topk-len(chosenid))])
        isnodlist = np.zeros([topk])

        
        for i,id in enumerate(chosenid):
            target = pbb[id,1:]
            isnod = pbb_label[id]
            crop,coord = self.crop(img,target)
            # print('-',crop.shape,coord.shape)#(1, 96, 96, 96) (3, 24, 24, 24)
            if self.phase=='train':
                crop,coord = augment(crop,coord,
                                 ifflip=self.augtype['flip'],ifrotate=self.augtype['rotate'],
                                ifswap = self.augtype['swap'])
            crop = crop.astype(np.float32)
            croplist[i] = crop
            coordlist[i] = coord
            isnodlist[i] = isnod
            
        if self.phase!='test':
            y = np.array([self.yset[idx]])
            # print('-',isnodlist, y)#[1. 0. 0. 0. 0.] [[1]]
            # print('-',croplist.shape, torch.from_numpy(coordlist).shape, torch.from_numpy(isnodlist).shape, torch.from_numpy(y).shape)#(5, 1, 96, 96, 96) torch.Size([5, 3, 24, 24, 24]) torch.Size([5]) torch.Size([1, 1])
            return torch.from_numpy(croplist).float(), torch.from_numpy(coordlist).float(), torch.from_numpy(isnodlist).int(), torch.from_numpy(y)
        else:
            return torch.from_numpy(croplist).float(), torch.from_numpy(coordlist).float(), torch.from_numpy(isnodlist).int()
    def __len__(self):
        if self.phase != 'test':
            return len(self.candidate_box)
        else:
            return len(self.candidate_box)
        

        
class simpleCrop():
    def __init__(self,config,phase):
        self.crop_size = config['crop_size']#[96,96,96]
        self.scaleLim = config['scaleLim']#[0.85,1.15]
        self.radiusLim = config['radiusLim']#[6,100]
        self.jitter_range = config['jitter_range']#0.15
        self.isScale = config['augtype']['scale'] and phase=='train'#False
        self.stride = config['stride']#4
        self.filling_value = config['filling_value']#160
        self.phase = phase
        
    def __call__(self,imgs,target):
        if self.isScale:
            radiusLim = self.radiusLim#[6,100]
            scaleLim = self.scaleLim#[0.85,1.15]
            scaleRange = [np.min([np.max([(radiusLim[0]/target[3]),scaleLim[0]]),1])
                         ,np.max([np.min([(radiusLim[1]/target[3]),scaleLim[1]]),1])]
            scale = np.random.rand()*(scaleRange[1]-scaleRange[0])+scaleRange[0]
            crop_size = (np.array(self.crop_size).astype('float')/scale).astype('int')
        else:
            crop_size = np.array(self.crop_size).astype('int')
        if self.phase=='train':
            jitter_range = target[3]*self.jitter_range
            jitter = (np.random.rand(3)-0.5)*jitter_range
        else:
            jitter = 0
        start = (target[:3]- crop_size/2 + jitter).astype('int')
        pad = [[0,0]]
        for i in range(3):
            if start[i]<0:
                leftpad = -start[i]
                start[i] = 0
            else:
                leftpad = 0
            if start[i]+crop_size[i]>imgs.shape[i+1]:
                rightpad = start[i]+crop_size[i]-imgs.shape[i+1]
            else:
                rightpad = 0
            pad.append([leftpad,rightpad])
        imgs = np.pad(imgs,pad,'constant',constant_values =self.filling_value)
        crop = imgs[:,start[0]:start[0]+crop_size[0],start[1]:start[1]+crop_size[1],start[2]:start[2]+crop_size[2]]
        
        normstart = np.array(start).astype('float32')/np.array(imgs.shape[1:])-0.5
        normsize = np.array(crop_size).astype('float32')/np.array(imgs.shape[1:])
        xx,yy,zz = np.meshgrid(np.linspace(normstart[0],normstart[0]+normsize[0],self.crop_size[0]/self.stride),
                           np.linspace(normstart[1],normstart[1]+normsize[1],self.crop_size[1]/self.stride),
                           np.linspace(normstart[2],normstart[2]+normsize[2],self.crop_size[2]/self.stride),indexing ='ij')
        coord = np.concatenate([xx[np.newaxis,...], yy[np.newaxis,...],zz[np.newaxis,:]],0).astype('float32')

        if self.isScale:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                crop = zoom(crop,[1,scale,scale,scale],order=1)
            newpad = self.crop_size[0]-crop.shape[1:][0]
            if newpad<0:
                crop = crop[:,:-newpad,:-newpad,:-newpad]
            elif newpad>0:
                pad2 = [[0,0],[0,newpad],[0,newpad],[0,newpad]]
                crop = np.pad(crop,pad2,'constant',constant_values =self.filling_value)

        return crop,coord#(1, 96, 96, 96) (3, 24, 24, 24)

def sample(conf,N,T=1):
    if len(conf)>N:
        # print('len(conf)',len(conf))
        target = [i for i in range(len(conf))]
        chosen_list = []
        for i in range(N):
            chosenidx = sampleone(target,conf,T)
            chosen_list.append(target[chosenidx])
            # print('target',target, chosenidx)
            target.pop(chosenidx)
            conf = np.delete(conf, chosenidx)

            
        return chosen_list
    else:
        return np.arange(len(conf))

def sampleone(target,conf,T):
    assert len(conf)>1
    p = softmax(conf/T)
    p = np.max([np.ones_like(p)*0.00001,p],axis=0)
    p = p/np.sum(p)
    # print(len(target),p)
    return np.random.choice(np.arange(len(target)),size=1,replace = False, p=p)[0]

def softmax(x):
    maxx = np.max(x)
    return np.exp(x-maxx)/np.sum(np.exp(x-maxx))


def augment(sample, coord, ifflip = True, ifrotate=True, ifswap = True):
    #                     angle1 = np.random.rand()*180
    if ifrotate:
        validrot = False
        counter = 0
        angle1 = np.random.rand()*180
        size = np.array(sample.shape[2:4]).astype('float')
        rotmat = np.array([[np.cos(angle1/180*np.pi),-np.sin(angle1/180*np.pi)],[np.sin(angle1/180*np.pi),np.cos(angle1/180*np.pi)]])
        sample = rotate(sample,angle1,axes=(2,3),reshape=False)
    if ifswap:
        if sample.shape[1]==sample.shape[2] and sample.shape[1]==sample.shape[3]:
            axisorder = np.random.permutation(3)
            sample = np.transpose(sample,np.concatenate([[0],axisorder+1]))
            coord = np.transpose(coord,np.concatenate([[0],axisorder+1]))
            
    if ifflip:
        flipid = np.array([np.random.randint(2),np.random.randint(2),np.random.randint(2)])*2-1
        sample = np.ascontiguousarray(sample[:,::flipid[0],::flipid[1],::flipid[2]])
        coord = np.ascontiguousarray(coord[:,::flipid[0],::flipid[1],::flipid[2]])
    return sample, coord 
