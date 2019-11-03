import os
import shutil
import numpy as np
from config_training import config
from scipy.io import loadmat
import numpy as np
import h5py
import csv
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
# sys.path.append('../preprocessing')
from step1_GGO import step1_python
import warnings
warnings.filterwarnings('ignore')
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
    ct_lungsegpath = './Data/ForTest/TestLungSeg/'
    ct_segpath = './Data/ForTest/TestSeg/'
    
    resolution = np.array([1,1,1])
    name = filelist[id].split('/')[-1].split('.')[0]
    label = annos[annos[:,0]==name]
    label = label[:,[3,1,2,4]].astype('float')#zxyd
    print('id',id, name, label) 
    im, m1, m2, spacing, origin = step1_python(os.path.join(data_path,name))#preprocess
    Mask = m1+m2
    # print('Mask.shape',Mask.shape)#(104, 512, 512)
    newshape = np.round(np.array(Mask.shape)*spacing/resolution)
    # print('newshape',newshape)#[260. 350. 350.]
    xx,yy,zz= np.where(Mask)
    print('xx,yy,zz',xx,yy,zz)
    box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
    print('box0',box, box.shape)#[[ 30. 235.][85.44924855 287.79306912][42.38282728 307.61729479]] (3, 2)
    box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)#lung box np.expand_dims:(3,)to (3, 1)
    print('box1',box, box.shape)
    Nodulemask,Nodulemaskorigin,Nodulemaskspacing,Nodulemaskisflip = load_itk_image(os.path.join(ct_segpath,name+'.mhd'))
    # print('box1',Nodulemask.shape, m1.shape)#[[45. 287.5][82.265625 303.046875][27.421875 336.796875]] (3, 2)
    # 
    box = np.floor(box).astype('int')
    print('box2',box, box.shape)
    margin = 5
    extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T
    extendbox = extendbox.astype('int')
    print('extendbox',extendbox, extendbox.shape)
    convex_mask = m1
    dm1 = process_mask(m1)#(凸包&扩张)
    dm2 = process_mask(m2)
    dilatedMask = dm1+dm2
    Mask = m1+m2
    extramask = dilatedMask ^ Mask
    bone_thresh = 210
    pad_value = 170
    im[np.isnan(im)]=-2000
    sliceim = lumTrans(im)#归一化到0-255
    sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
    bones = sliceim*extramask>bone_thresh
    sliceim[bones] = pad_value
    sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
    Mask1,_ = resample(dilatedMask,spacing,resolution,order=1)
    NoduleMask1,_ = resample(Nodulemask,spacing,resolution,order=1)
    sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],extendbox[1,0]:extendbox[1,1],extendbox[2,0]:extendbox[2,1]]
    Mask2 = Mask1[extendbox[0,0]:extendbox[0,1],extendbox[1,0]:extendbox[1,1],extendbox[2,0]:extendbox[2,1]]
    NoduleMask2 = NoduleMask1[extendbox[0,0]:extendbox[0,1],extendbox[1,0]:extendbox[1,1],extendbox[2,0]:extendbox[2,1]]
    sliceim = sliceim2[np.newaxis,...]
    Mask = Mask2[np.newaxis,...]
    NoduleMask = NoduleMask2[np.newaxis,...]
    # np.save(os.path.join(ct_lungsegpath,name+'_Lungmask.npy'),Mask)
    # np.save(os.path.join(ct_lungsegpath,name+'_Nodulemask.npy'),NoduleMask)
    # np.save(os.path.join(prep_folder,name+'_clean.npy'),sliceim)
    # np.save(os.path.join(prep_folder, name+'_spacing.npy'), spacing)#float
    # np.save(os.path.join(prep_folder, name+'_extendbox.npy'), extendbox)#int
    # np.save(os.path.join(prep_folder, name+'_origin.npy'), origin)#float
    print('save--clean--Lungmask--', name)
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

    # print(name)
    
def prepare_GGO():
    prep_folder = '/home/zhaojie/zhaojie/Lung/DSB_Code/DSB2017-master/training/Data/ForTest/Preprocess/'
    ct_rawpath = '/home/zhaojie/zhaojie/Lung/DSB_Code/DSB2017-master/training/Data/GGO_data/'
    TestLable = './Data/ForTest/TestLable.csv'
    ct_testpath = './Data/ForTest/TestSet/'
    ct_segpath = './Data/ForTest/TestSeg/'
    ct_lungsegpath = './Data/ForTest/TestLungSeg/'
    SaveAndRename = False
    DO = True
    Preprocess = True
    rows = []
    filedirs = [os.path.join(ct_rawpath,f) for f in os.listdir(ct_rawpath) if f.endswith('img.nii') ]
    namelist = [f.split('_')[0] for f in os.listdir(ct_rawpath) if f.endswith('img.nii') ]
    if SaveAndRename:
        print('start changing luna name')
        
        namesnpy = np.array(namelist)
        np.save(os.path.join('./Data/ForTest',"testnames.npy"),namesnpy)
        print('namelist',namelist)
        if not os.path.exists(ct_testpath):
            os.mkdir(ct_testpath)
        
        for f in filedirs:
            filename = f.split('/')[-1]
            segname = f.replace('img','mask')
            # print('filename',filename)
            if DO and 0>1:
                shutil.copy(os.path.join(ct_rawpath,filename),os.path.join(ct_testpath,filename))
            ct = sitk.ReadImage(os.path.join(ct_rawpath,filename),sitk.sitkInt16)
            ct_array = sitk.GetArrayFromImage(ct)
            seg = sitk.ReadImage(os.path.join(ct_rawpath,segname),sitk.sitkInt16)
            seg_array = sitk.GetArrayFromImage(seg)
            # print('ct_array.shape',ct_array.shape)
            # 坐标轴（z,y,x）,这样是dicom存储文件的格式，即第一个维度为z轴便于图片堆叠
            ArrayDicom = ct_array
            ArrayDicomSeg = seg_array
            # 将现在的numpy数组通过SimpleITK转化为mhd和raw文件
            sitk_img = sitk.GetImageFromArray(ArrayDicom, isVector=False)
            sitk_img.SetSpacing(ct.GetSpacing())
            sitk_img.SetOrigin(ct.GetOrigin())
            if DO:
                sitk.WriteImage(sitk_img, os.path.join(ct_testpath, filename.split('_')[0] + ".mhd"))
                # os.remove(os.path.join(ct_testpath,filename)
            print(os.path.join(ct_testpath, filename.split('_')[0] + ".mhd"))
            sitk_img = sitk.GetImageFromArray(ArrayDicomSeg, isVector=False)
            sitk_img.SetSpacing(ct.GetSpacing())
            sitk_img.SetOrigin(ct.GetOrigin())
            if DO:
                sitk.WriteImage(sitk_img, os.path.join(ct_segpath, filename.split('_')[0] + ".mhd"))
            if DO:
                headers = ['id','coordx1','coordx1','coordx1','diameter']
                rows.append([filename.split('_')[0],0,0,0,0])
                with open(TestLable,'w') as f:
                    f_csv = csv.writer(f)
                    f_csv.writerow(headers)
                    for i_rows in range(len(rows)):
                        f_csv.writerow(rows[i_rows])
        print('end changing name')
    
    if Preprocess:
        print('starting preprocessing')
        alllabelfiles = TestLable
        tmp = []
        content = np.array(pandas.read_csv(alllabelfiles))
        content = content[content[:,0]!=np.nan]
        tmp.append(content[:,:5])
        alllabel = np.concatenate(tmp,0)#id,x,y,z,d
        pool = Pool()
        filelist = [os.path.join(ct_testpath,f) for f in os.listdir(ct_testpath) if f.endswith('.mhd') ]#namelist
        partial_savenpy = partial(savenpy,annos= alllabel,filelist=filelist,data_path=ct_testpath,prep_folder=prep_folder)

        N = len(filelist)
            #savenpy(1)
        _=pool.map(partial_savenpy,range(N))
        pool.close()
        pool.join()
        print('end preprocessing')
if __name__=='__main__':

    prepare_GGO()#mv and rename luna data
    
