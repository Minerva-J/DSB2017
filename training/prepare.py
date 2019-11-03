import os
import shutil
import numpy as np
from config_training import config


from scipy.io import loadmat
import numpy as np
import h5py
import pandas
import scipy
from scipy.ndimage.interpolation import zoom
from skimage import measure
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image
import pandas
from multiprocessing import Pool
from functools import partial
import sys
sys.path.append('../preprocessing')
from step1 import step1_python
import warnings

def resample(imgs, spacing, new_spacing,order=2):
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)
        return imgs, true_spacing
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')
def worldToVoxelCoord(worldCoord, origin, spacing):
     
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        # print('transformM',transformM)
        if np.any( transformM!=np.array([1,0,0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
     
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     
    return numpyImage, numpyOrigin, numpySpacing,isflip

def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0:
            mask2 = convex_hull_image(mask1)#(凸包&扩张)
            if np.sum(mask2)>1.5*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3,1)  
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10) 
    return dilatedMask


def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg


def savenpy(id,annos,filelist,data_path,prep_folder):   
    print('id',id)    
    resolution = np.array([1,1,1])
    name = filelist[id]
    label = annos[annos[:,0]==name]
    label = label[:,[3,1,2,4]].astype('float')#zxyd
    
    im, m1, m2, spacing = step1_python(os.path.join(data_path,name))#preprocess
    Mask = m1+m2
    # print('Mask.shape',Mask.shape)#(104, 512, 512)
    newshape = np.round(np.array(Mask.shape)*spacing/resolution)
    # print('newshape',newshape)#[260. 350. 350.]
    xx,yy,zz= np.where(Mask)
    box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
    # print('box0',box, box.shape)#[[ 30. 235.][85.44924855 287.79306912][42.38282728 307.61729479]] (3, 2)
    box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)#lung box np.expand_dims:(3,)to (3, 1)
    # print('spacing',spacing, np.expand_dims(spacing,1))#[2.5 0.664062 0.664062] [2.5 ][0.664062][0.664062]]
    # print('box1',box, box.shape)#[[45. 287.5][82.265625 303.046875][27.421875 336.796875]] (3, 2)
    # 
    box = np.floor(box).astype('int')
    margin = 5
    extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T
    extendbox = extendbox.astype('int')



    convex_mask = m1
    dm1 = process_mask(m1)#(凸包&扩张)
    dm2 = process_mask(m2)
    dilatedMask = dm1+dm2
    Mask = m1+m2
    # print('---0',im.shape,Mask.shape)
    extramask = dilatedMask ^ Mask
    bone_thresh = 210
    pad_value = 170
    im[np.isnan(im)]=-2000
    sliceim = lumTrans(im)#归一化到0-255
    sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
    bones = sliceim*extramask>bone_thresh
    sliceim[bones] = pad_value
    # print('---1',im.shape,sliceim.shape,Mask.shape)
    sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
    Mask1,_ = resample(Mask,spacing,resolution,order=1)
    sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],extendbox[1,0]:extendbox[1,1],extendbox[2,0]:extendbox[2,1]]
    Mask2 = Mask1[extendbox[0,0]:extendbox[0,1],extendbox[1,0]:extendbox[1,1],extendbox[2,0]:extendbox[2,1]]
    sliceim = sliceim2[np.newaxis,...]
    Mask = Mask2[np.newaxis,...]
    # print('---1',im.shape,sliceim.shape,Mask.shape)
    print('save mask clean label to',prep_folder)
    np.save(os.path.join(prep_folder,name+'_mask.npy'),Mask)
    np.save(os.path.join(prep_folder,name+'_clean.npy'),sliceim)
    
    if len(label)==0:
        label2 = np.array([[0,0,0,0]])
    elif len(label[0])==0:
        label2 = np.array([[0,0,0,0]])
    elif label[0][0]==0:
        label2 = np.array([[0,0,0,0]])
    else:
        haslabel = 1
        label2 = np.copy(label).T
        label2[:3] = label2[:3][[0,2,1]]
        label2[:3] = label2[:3]*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
        label2[3] = label2[3]*spacing[1]/resolution[1]
        label2[:3] = label2[:3]-np.expand_dims(extendbox[:,0],1)
        label2 = label2[:4].T
    np.save(os.path.join(prep_folder,name+'_label.npy'),label2)

    print(name)

def full_prep(step1=True,step2 = True):
    warnings.filterwarnings("ignore")
    prep_folder = config['preprocess_result_path_with_mask']#/home/zhaojie/zhaojie/Lung/DSB_Code/DSB2017-Minerva/training/Data/preprocess_whth_mask/
    data_path = config['stage1_data_path']#/home/zhaojie/zhaojie/Lung/data/2017DataScienceBowl/stage1/stage1/'
    finished_flag = '.flag_prepkaggle'
    
    if not os.path.exists(finished_flag):
        alllabelfiles = config['stage1_annos_path']#['./detector/labels/label_job5.csv',
                # './detector/labels/label_job4_2.csv',
                # './detector/labels/label_job4_1.csv',
                # './detector/labels/label_job0.csv',
                # './detector/labels/label_qualified.csv']
        tmp = []
        for f in alllabelfiles:
            content = np.array(pandas.read_csv(f))
            content = content[content[:,0]!=np.nan]
            tmp.append(content[:,:5])
        alllabel = np.concatenate(tmp,0)#id,x,y,z,d
        filelist = os.listdir(config['stage1_data_path'])

        if not os.path.exists(prep_folder):
            os.mkdir(prep_folder)
        #eng.addpath('preprocessing/',nargout=0)

        print('starting preprocessing')
        pool = Pool()
        filelist = [f for f in os.listdir(data_path)]#1595个子文件
        partial_savenpy = partial(savenpy,annos= alllabel,filelist=filelist,data_path=data_path,prep_folder=prep_folder )

        N = len(filelist)
        _=pool.map(partial_savenpy,range(N))
        pool.close()
        pool.join()
        print('end preprocessing')
    # f= open(finished_flag,"w+")        

def savenpy_luna(id,annos,filelist,luna_segment,luna_data,savepath):
    islabel = True
    isClean = True
    resolution = np.array([1,1,1])
#     resolution = np.array([2,2,2])
    name = filelist[id]
    
    # if not os.path.exists(os.path.join(savepath,name+'_clean.npy')):
    if 1>0:
        # print('0',os.path.join(luna_segment,name+'.mhd'))
        Mask,origin,spacing,isflip = load_itk_image(os.path.join(luna_segment,name+'.mhd'))
        if isflip:
            Mask = Mask[:,::-1,::-1]
        newshape = np.round(np.array(Mask.shape)*spacing/resolution).astype('int')
        m1 = Mask==3
        m2 = Mask==4
        Mask = m1+m2
        
        xx,yy,zz= np.where(Mask)
        box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
        box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
        box = np.floor(box).astype('int')
        margin = 5
        extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T
        
        this_annos = np.copy(annos[annos[:,0]==int(name)])        
        
        if isClean:
            convex_mask = m1
            dm1 = process_mask(m1)
            dm2 = process_mask(m2)
            dilatedMask = dm1+dm2
            Mask = m1+m2
            extramask = dilatedMask ^ Mask
            bone_thresh = 210
            pad_value = 170
        
            sliceim,origin,spacing,isflip = load_itk_image(os.path.join(luna_data,name+'.mhd'))
            if isflip:
                sliceim = sliceim[:,::-1,::-1]
                print('flip!')
            sliceim = lumTrans(sliceim)
            sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
            bones = (sliceim*extramask)>bone_thresh
            sliceim[bones] = pad_value
            
            sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
            Mask1,_ = resample(Mask,spacing,resolution,order=1)
            sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                        extendbox[1,0]:extendbox[1,1],
                        extendbox[2,0]:extendbox[2,1]]
            Mask2 = Mask1[extendbox[0,0]:extendbox[0,1],
                        extendbox[1,0]:extendbox[1,1],
                        extendbox[2,0]:extendbox[2,1]]
            sliceim = sliceim2[np.newaxis,...]
            Mask3 = Mask2[np.newaxis,...]
            np.save(os.path.join(savepath,name+'_clean.npy'),sliceim)
            np.save(os.path.join(savepath,name+'_mask.npy'),Mask3)
            # print('save clean mask lable to',savepath)
        
        if islabel:
        
            this_annos = np.copy(annos[annos[:,0]==int(name)])
            label = []
            if len(this_annos)>0:
                
                for c in this_annos:
                    pos = worldToVoxelCoord(c[1:4][::-1],origin=origin,spacing=spacing)
                    if isflip:
                        pos[1:] = Mask.shape[1:3]-pos[1:]
                    label.append(np.concatenate([pos,[c[4]/spacing[1]]]))
                
            label = np.array(label)
            if len(label)==0:
                label2 = np.array([[0,0,0,0]])
            else:
                label2 = np.copy(label).T
                label2[:3] = label2[:3]*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
                label2[3] = label2[3]*spacing[1]/resolution[1]
                label2[:3] = label2[:3]-np.expand_dims(extendbox[:,0],1)
                label2 = label2[:4].T
            np.save(os.path.join(savepath,name+'_label.npy'),label2)
            
        print(name)

def preprocess_luna():
    luna_segment = config['luna_segment']#'/home/zhaojie/zhaojie/Lung/DSB_Code/DSB2017-Minerva/training/Data/seg-lungs-LUNA16/'
    savepath = config['preprocess_result_path_with_mask']#'/home/zhaojie/zhaojie/Lung/DSB_Code/DSB2017-Minerva/training/Data/preprocess_whth_mask/'
    luna_data = config['luna_data']#'/home/zhaojie/zhaojie/Lung/DSB_Code/DSB2017-Minerva/training/Data/allset/'
    luna_label = config['luna_label']#'./detector/labels/lunaqualified.csv',
    finished_flag = '.flag_preprocessluna'
    print('starting preprocessing luna')
    if not os.path.exists(finished_flag):
        filelist = [f.split('.mhd')[0] for f in os.listdir(luna_data) if f.endswith('.mhd') ]
        annos = np.array(pandas.read_csv(luna_label))

        if not os.path.exists(savepath):
            os.mkdir(savepath)

        
        pool = Pool()
        partial_savenpy_luna = partial(savenpy_luna,annos=annos,filelist=filelist,luna_segment=luna_segment,luna_data=luna_data,savepath=savepath)

        N = len(filelist)
        #savenpy(1)
        _=pool.map(partial_savenpy_luna,range(N))
        pool.close()
        pool.join()
    print('end preprocessing luna')
    # f= open(finished_flag,"w+")
    
def prepare_luna():
    print('start changing luna name')
    luna_raw = config['luna_raw']#'/home/zhaojie/zhaojie/Lung/DSB_Code/DSB2017-master/training/Data/subset_data/'
    luna_abbr = config['luna_abbr']#'./detector/labels/shorter.csv'
    luna_data = config['luna_data']#'/home/zhaojie/zhaojie/Lung/DSB_Code/DSB2017-Minerva/training/Data/allset/'
    luna_segment = config['luna_segment']#'/home/zhaojie/zhaojie/Lung/DSB_Code/DSB2017-Minerva/training/Data/seg-lungs-LUNA16/'
    finished_flag = '.flag_prepareluna0'
    
    if True:
        
        subsetdirs = [os.path.join(luna_raw,f) for f in os.listdir(luna_raw) if f.startswith('subset') and os.path.isdir(os.path.join(luna_raw,f))]
        if not os.path.exists(luna_data):
            os.mkdir(luna_data)
        
        abbrevs = np.array(pandas.read_csv(config['luna_abbr'],header=None))
        namelist = list(abbrevs[:,1])
        ids = abbrevs[:,0]
        # print('-----------------', subsetdirs)
        for d in subsetdirs:
            files = os.listdir(d)
            
            files.sort()
            for f in files:
                name = f[:-4]
                
                id = ids[namelist.index(name)]
                print(name,id)
                filename = '0'*(3-len(str(id)))+str(id)#3位数字
                shutil.move(os.path.join(d,f),os.path.join(luna_data,filename+f[-4:]))
                print('0shutil.move',os.path.join(d,f),os.path.join(luna_data,filename+f[-4:]))

        files = [f for f in os.listdir(luna_data) if f.endswith('mhd')]
        for file in files:
            with open(os.path.join(luna_data,file),'r') as f:
                content = f.readlines()
                id = file.split('.mhd')[0]
                filename = '0'*(3-len(str(id)))+str(id)
                content[-1]='ElementDataFile = '+filename+'.raw\n'
            with open(os.path.join(luna_data,file),'w') as f:
                f.writelines(content)

                
        seglist = os.listdir(luna_segment)
        for f in seglist:
            if f.endswith('.mhd'):

                name = f[:-4]
                lastfix = f[-4:]
            else:
                name = f[:-5]
                lastfix = f[-5:]
            if name in namelist:
                id = ids[namelist.index(name)]
                filename = '0'*(3-len(str(id)))+str(id)

                shutil.move(os.path.join(luna_segment,f),os.path.join(luna_segment,filename+lastfix))
                print('1shutil.move',os.path.join(luna_segment,f),os.path.join(luna_segment,filename+lastfix))
                # os.remove(os.path.join(luna_segment,f))

        files = [f for f in os.listdir(luna_segment) if f.endswith('mhd')]
        for file in files:
            with open(os.path.join(luna_segment,file),'r') as f:
                content = f.readlines()
                id =  file.split('.mhd')[0]
                filename = '0'*(3-len(str(id)))+str(id)
                content[-1]='ElementDataFile = '+filename+'.zraw\n'
                # print('1content[-1]',content[-1])
            with open(os.path.join(luna_segment,file),'w') as f:
                f.writelines(content)
    print('end changing luna name')
    # f= open(finished_flag,"w+")
    
if __name__=='__main__':
    # full_prep(step1=True,step2=True)#preprocess DSB data
    # prepare_luna()#mv and rename luna data
    preprocess_luna()#preprocess luna data
    
