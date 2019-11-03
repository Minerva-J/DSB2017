import argparse
import os
import time
import numpy as np
from importlib import import_module
import shutil
import sys
from split_combine import SplitComb

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

from layers import acc
from trainval_detector import *
from trainval_classifier import *
from data_detector import DataBowl3Detector
from data_classifier import DataBowl3Classifier

from utils import *

parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')
parser.add_argument('--model1', '-m1', metavar='MODEL', default='base',
                    help='model')
parser.add_argument('--model2', '-m2', metavar='MODEL', default='base',
                    help='model')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=None, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('-b2', '--batch-size2', default=3, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--save-freq', default='5', type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir', default='', type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')
parser.add_argument('--test1', default=0, type=int, metavar='TEST',
                    help='do detection test')
parser.add_argument('--test2', default=0, type=int, metavar='TEST',
                    help='do classifier test')
parser.add_argument('--test3', default=0, type=int, metavar='TEST',
                    help='do classifier test')
parser.add_argument('--split', default=8, type=int, metavar='SPLIT',
                    help='In the test phase, split the image to 8 parts')
parser.add_argument('--gpu', default='all', type=str, metavar='N',
                    help='use gpu')
parser.add_argument('--n_test', default=8, type=int, metavar='N',
                    help='number of gpu for test')
parser.add_argument('--debug', default=0, type=int, metavar='TEST',
                    help='debug mode')
parser.add_argument('--freeze_batchnorm', default=0, type=int, metavar='TEST',
                    help='freeze the batchnorm when training')

def main():
    global args
    args = parser.parse_args()
    
    
    torch.manual_seed(0)
    
    
    ##################################

    nodmodel = import_module(args.model1)
    config1, nod_net, loss, get_pbb = nodmodel.get_model()#net_detector_3
    args.lr_stage = config1['lr_stage']#np.array([50,100,140,160])
    args.lr_preset = config1['lr']#[0.01,0.001,0.0001,0.00001]

    
    save_dir = args.save_dir

    
    ##################################
    
    casemodel = import_module(args.model2)#net_classifier_3
    
    config2 = casemodel.config
    args.lr_stage2 = config2['lr_stage']#np.array([50,100,140,160])
    args.lr_preset2 = config2['lr']#[0.01,0.001,0.0001,0.00001]
    topk = config2['topk']#5
    case_net = casemodel.CaseNet(topk = topk,nodulenet=nod_net)

    args.miss_ratio = config2['miss_ratio']#1
    args.miss_thresh = config2['miss_thresh']#0.03
    if args.debug:
        args.save_dir = 'debug'
    
    ###################################
    
    
    
    
    
    
    ################################
    start_epoch = args.start_epoch#30
    if args.resume:
        print('resume',args.resume)
        checkpoint = torch.load(args.resume)
        if start_epoch == 0:
            start_epoch = checkpoint['epoch'] + 1
        if not save_dir:
            save_dir = checkpoint['save_dir']
        else:
            save_dir = os.path.join('results',save_dir)
        case_net.load_state_dict(checkpoint['state_dict'])
    else:
        if start_epoch == 0:
            start_epoch = 1
        if not save_dir:
            exp_id = time.strftime('%Y%m%d-%H%M%S', time.localtime())
            save_dir = os.path.join('results', args.model1 + '-' + exp_id)
        else:
            save_dir = os.path.join('results',save_dir)
    if args.epochs == None:
        end_epoch = args.lr_stage2[-1]#160
    else:
        end_epoch = args.epochs
    ################################
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logfile = os.path.join(save_dir,'log')
    if args.test1!=1 and args.test2!=1 :
        sys.stdout = Logger(logfile)
        pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
        for f in pyfiles:
            shutil.copy(f,os.path.join(save_dir,f))
    ################################
    torch.cuda.set_device(0)
    #nod_net = nod_net.cuda()
    case_net = case_net.cuda()
    loss = loss.cuda()
    cudnn.benchmark = True
    if not args.debug:
        case_net = DataParallel(case_net)
        nod_net = DataParallel(nod_net)
    ################################


    if args.test1 == 1:
        testsplit = np.load('full.npy')
        dataset = DataBowl3Classifier(testsplit, config2, phase = 'test')
        predlist = test_casenet(case_net,dataset).T
        anstable = np.concatenate([[testsplit],predlist],0).T
        df = pandas.DataFrame(anstable)
        df.columns={'id','cancer'}
        df.to_csv('allstage1.csv',index=False)
        return

    if args.test2 ==1:

        testsplit = np.load('test.npy')
        dataset = DataBowl3Classifier(testsplit, config2, phase = 'test')
        predlist = test_casenet(case_net,dataset).T
        anstable = np.concatenate([[testsplit],predlist],0).T
        df = pandas.DataFrame(anstable)
        df.columns={'id','cancer'}
        df.to_csv('quick',index=False)
        return
    if args.test3 == 1:
        testsplit3 = np.load('stage2.npy')
        dataset = DataBowl3Classifier(testsplit3,config2,phase = 'test')
        predlist = test_casenet(case_net,dataset).T
        anstable = np.concatenate([[testsplit3],predlist],0).T
        df = pandas.DataFrame(anstable)
        df.columns={'id','cancer'}
        df.to_csv('stage2_ans.csv',index=False)
        return
    print('save_dir',save_dir)
    print('save_freq',args.save_freq)
    print('----------Loading Data')
    trainsplit = np.load('kaggleluna_full.npy')
    valsplit = np.load('valsplit.npy')
    testsplit = np.load('test.npy')
    ##############################################
    dataset = DataBowl3Detector(trainsplit,config1,phase = 'train')
    train_loader_nod = DataLoader(dataset,batch_size = args.batch_size,
        shuffle = True,num_workers = args.workers,pin_memory=True)#kaggleluna_full

    dataset = DataBowl3Detector(valsplit,config1,phase = 'val')
    val_loader_nod = DataLoader(dataset,batch_size = args.batch_size,
        shuffle = False,num_workers = args.workers,pin_memory=True)#valsplit
    iter1, iter2, iter3 = next(iter(train_loader_nod))
    # print('len(train_loader_nod)',len(train_loader_nod))#1881
    # print('len(val_loader_nod)',len(val_loader_nod))#216
    # print("iter1: ", iter1.size())#([1, 1, 128, 128, 128])
    # print("iter2: ", iter2.size())#([1, 32, 32, 32, 3, 5])
    # print("iter3: ", iter3.size())#([1, 3, 32, 32, 32])
    optimizer = torch.optim.SGD(nod_net.parameters(), args.lr,momentum = 0.9,weight_decay = args.weight_decay)
    #########################################
    trainsplit = np.load('full.npy')
    dataset = DataBowl3Classifier(trainsplit,config2,phase = 'train')
    train_loader_case = DataLoader(dataset,batch_size = args.batch_size2,
        shuffle = True,num_workers = args.workers,pin_memory=True)#full
    
    dataset = DataBowl3Classifier(valsplit,config2,phase = 'val')
    val_loader_case = DataLoader(dataset,batch_size = max([args.batch_size2,1]),
        shuffle = False,num_workers = args.workers,pin_memory=True)#valsplit

    dataset = DataBowl3Classifier(trainsplit,config2,phase = 'val')
    all_loader_case = DataLoader(dataset,batch_size = max([args.batch_size2,1]),
        shuffle = False,num_workers = args.workers,pin_memory=True)#full
    iter1, iter2, iter3, iter4 = next(iter(train_loader_case))
    # print('len(train_loader_case)',len(train_loader_case))#1595
    # print('len(val_loader_case)',len(val_loader_case))#350
    # print("iter1: ", iter1.size())#([1, 5, 1, 96, 96, 96])
    # print("iter2: ", iter2.size())#([1, 5, 3, 24, 24, 24])
    # print("iter3: ", iter3.size())#([1, 5])isnodlist
    # print("iter4: ", iter4.size())#([1, 1, 1])
    optimizer2 = torch.optim.SGD(case_net.parameters(),
        args.lr,momentum = 0.9,weight_decay = args.weight_decay)
    ###############################################
    for epoch in range(start_epoch, end_epoch + 1):
        if epoch ==start_epoch:#30
            print('Epoch-train_casenet',epoch)
            lr = args.lr#1e-2
            debug = args.debug
            args.lr = 0.0
            args.debug = True
            train_casenet(epoch,case_net,train_loader_case,optimizer2,args)
            args.lr = lr
            args.debug = debug
            # print(stop)
        if epoch<args.lr_stage[-1]:#[50,100,140,160]160
            print('Epoch-train_nodulenet',epoch)
            train_nodulenet(train_loader_nod, nod_net, loss, epoch, optimizer, args)
            validate_nodulenet(val_loader_nod, nod_net, loss)
        if epoch>config2['startepoch']:#20
            print('Epoch-train_casenet',epoch)
            train_casenet(epoch,case_net,train_loader_case,optimizer2,args)
            val_casenet(epoch,case_net,val_loader_case,args)
            val_casenet(epoch,case_net,all_loader_case,args)

        if epoch % args.save_freq == 0:            
            state_dict = case_net.module.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()

            # torch.save({
                # 'epoch': epoch,
                # 'save_dir': save_dir,
                # 'state_dict': state_dict,
                # 'args': args},
                # os.path.join(save_dir, '%03d.ckpt' % epoch))
if __name__ == '__main__':
    main()

