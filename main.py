# from preprocessing import full_prep
from full_prep import full_prep
from config_submit import config as config_submit

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

from layers import acc
from data_detector import DataBowl3Detector,collate
from data_classifier import DataBowl3Classifier

from utils import *
from split_combine import SplitComb
from test_detect import test_detect
from importlib import import_module
import pandas

datapath = config_submit['datapath']#'/home/zhaojie/zhaojie/Lung/data/2017DataScienceBowl/stage2/stage2/'
prep_result_path = config_submit['preprocess_result_path']#'./prep_result/'
skip_prep = config_submit['skip_preprocessing']#False
skip_detect = config_submit['skip_detect']#False
print('-----------------')
if not skip_prep:
    testsplit = full_prep(datapath,prep_result_path,n_worker =None,use_existing=True)
else:
    testsplit = os.listdir(datapath)

nodmodel = import_module(config_submit['detector_model'].split('.py')[0])#'net_detector'
config1, nod_net, loss, get_pbb = nodmodel.get_model()
checkpoint = torch.load(config_submit['detector_param'])#'./model/detector.ckpt'
nod_net.load_state_dict(checkpoint['state_dict'])

torch.cuda.set_device(0)
nod_net = nod_net.cuda()
cudnn.benchmark = True
nod_net = DataParallel(nod_net)

bbox_result_path = './bbox_result'
if not os.path.exists(bbox_result_path):
    os.mkdir(bbox_result_path)
#testsplit = [f.split('_clean')[0] for f in os.listdir(prep_result_path) if '_clean' in f]

if not skip_detect:
    margin = 32
    # sidelen = 144
    sidelen = 128
    config1['datadir'] = prep_result_path
    split_comber = SplitComb(sidelen,config1['max_stride'],config1['stride'],margin,pad_value= config1['pad_value'])#144,16,4,32,170

    dataset = DataBowl3Detector(testsplit,config1,phase='test',split_comber=split_comber)
    test_loader = DataLoader(dataset,batch_size = 1,
        shuffle = False,num_workers = 32,pin_memory=False,collate_fn =collate)

    # test_detect(test_loader, nod_net, get_pbb, bbox_result_path,config1,n_gpu=config_submit['n_gpu'])
    iter1, iter2, iter3, iter4 = next(iter(test_loader))
    print('len(test_loader)',len(test_loader))#506
    print("iter1: ", len(iter1))#1
    print("iter2: ", len(iter2))#1
    print("iter3: ", len(iter3))
    print("iter4: ", iter4)
    


casemodel = import_module(config_submit['classifier_model'].split('.py')[0])#'net_classifier'
casenet = casemodel.CaseNet(topk=5)
config2 = casemodel.config
checkpoint = torch.load(config_submit['classifier_param'])#'./model/classifier.ckpt'
casenet.load_state_dict(checkpoint['state_dict'])

torch.cuda.set_device(0)
casenet = casenet.cuda()
cudnn.benchmark = True
casenet = DataParallel(casenet)

filename = config_submit['outputfile']#'prediction.csv'



def test_casenet(model,testset):
    data_loader = DataLoader(
        testset,
        batch_size = 1,
        shuffle = False,
        num_workers = 32,
        pin_memory=True)
    #model = model.cuda()
    model.eval()
    predlist = []
    
    #     weight = torch.from_numpy(np.ones_like(y).float().cuda()
    for i,(x,coord) in enumerate(data_loader):

        coord = Variable(coord).cuda()
        x = Variable(x).cuda()
        nodulePred,casePred,_ = model(x,coord)
        predlist.append(casePred.data.cpu().numpy())
        #print([i,data_loader.dataset.split[i,1],casePred.data.cpu().numpy()])
    predlist = np.concatenate(predlist)
    return predlist    
config2['bboxpath'] = bbox_result_path#'./bbox_result'
config2['datadir'] = prep_result_path#'./prep_result/'



dataset = DataBowl3Classifier(testsplit, config2, phase = 'test')
predlist = test_casenet(casenet,dataset).T
df = pandas.DataFrame({'id':testsplit, 'cancer':predlist})
df.to_csv(filename,index=False)
