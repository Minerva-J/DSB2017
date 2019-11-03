# -- coding:utf-8 --
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
def nms(output, nms_th):
    if len(output) == 0:
        return output
    output = output[np.argsort(-output[:, 0])]
    bboxes = [output[0]]
    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j in range(len(bboxes)):
            if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th:
                flag = -1
                break
        if flag == 1:
            bboxes.append(bbox)
    bboxes = np.asarray(bboxes, np.float32)
    return bboxes
	
def iou(box0, box1):
    r0 = box0[3] / 2#半径
    s0 = box0[:3] - r0#xyz的左上角
    e0 = box0[:3] + r0#xyz的右上角
    r1 = box1[3] / 2
    s1 = box1[:3] - r1#xyz的左上角
    e1 = box1[:3] + r1#xyz的右上角
    overlap = []
    for i in range(len(s0)):#3个
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))#最小的右下角-最大的左下角=((x1-x0),(y1-y0),(z1-z0))
    intersection = overlap[0] * overlap[1] * overlap[2]#(x1-x0)*(y1-y0)*(z1-z0)
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection#体积就是3个直径的乘积
    return intersection / union
nmsthresh = 0.1
detp = -1
data_dir = '/home/zhaojie/zhaojie/Lung/DSB_Code/DSB2017-master/training/detector/results/res18/TestBbox/'
filenames = os.path.join(data_dir, 'T000807321_pbb.npy')
pbb = np.load(filenames, mmap_mode='r')
# imgs = np.load('/home/zhaojie/zhaojie/Lung/code/detector_py3/results/dpn3d26/retrft960/train962/1.3.6.1.4.1.14519.5.2.1.6279.6001.102681962408431413578140925249_pbb.npy')
print('------0',pbb.shape)
# print('------0',pbb)
# pbbold = np.array(pbb[pbb[:,0] > detp])#根据阈值过滤掉概率低的
# pbbold = np.array(pbbold[pbbold[:,-1] > 3])#根据半径过滤掉小于3mm的

# pbbold = pbbold[np.argsort(-pbbold[:,0])][:1000] #取概率值前1000的结节作为输出，不然直接进行nms耗时太长
pbb = nms(pbb, 0.2)#对输出的结节进行nms
# pbb = nms(pbbold, nmsthresh)#对输出的结节进行nms
print('len(pbb)',pbb.shape,pbb)
# for i in range(imgs.shape[0]):
    # name = str(i) + '_' + '.png'
    # cv2.imwrite(os.path.join('./CT_cleannp/', name), imgs[i])
	
	
