# DSB2017 PyTorch1.0 Python3.7 with multi GPUs

# Illustration
The project is based on lfz's work to upgrade and modify the version, and add some additional visualization functions.The original version is applicable to python 2.7 and pytorch 0.1 from https://github.com/lfz/DSB2017, and the Paper link is
https://arxiv.org/abs/1711.08324
Please cite this paper if you find this project helpful for your research.膜拜大佬!!!

# Dependencies

python 3.7, CUDA 10.1,SimpleITK 1.2.2, numpy 1.17, matplotlib 3.1.1, scikit-image (0.21), scipy (0.3.1), pyparsing (2.4.2), pytorch (1.0) (anaconda is recommended)
other packages could be the latest version.

# Instructions for runing

# Training:

# 1.Install all dependencies

# 2.Download dataset:

Download luna data from https://luna16.grand-challenge.org/download/, and DSB data from https://www.kaggle.com/c/data-science-bowl-2017/data,

# 3.Prepare data:

Prepare stage1 data, LUNA data, and LUNA segment results  unzip them to different folders('stage1_data_path', 'luna_raw', 'luna_segment') from ./training/config_training.py
Go to ./training and open config_training.py
Filling in stage1_data_path, luna_raw, luna_segment with the path mentioned above
Filling in luna_data, preprocess_result_path, with tmp folders

# 4.Proprocess data：

cd ./training/ and python prepare.py

4.1 The function of full_prep in 381 line proproces DSB data stage1, and generate mask.npy clean.npy labe.npy to folder of config['preprocess_result_path_with_mask']. The files number is 1595 * 3=4785.

4.2 The function of prepare_luna in 382 line copy and rename luna data.Copy luna data from config['luna_raw'] to config['luna_data'], and rename all files to like 001.mhd and 001.raw. The folder of config['luna_data'] contain 888*2 = 1776 files. Copy luna seg data from config['luna_segment'] to config['luna_segment'], and rename all files to like 001.mhd and 001.zraw. The folder of config['luna_segment'] contain 888 * 2 = 1776 files.

4.3 The function of preprocess_luna in 383 line proproces luna data, and generate mask.npy clean.npy labe.npy to folder of config['preprocess_result_path_with_mask']. The files number is 2664.
After runing prepare.py, The folder of config['preprocess_result_path_with_mask'] contains 7449(1595 * 3+888 * 3) files.
# 5.Run detector：

cd ./detector and python main.py --model res18 -b 12 --epochs 1000 --save-dir res18/CkptFile
You can modify -b(batch_size) depend on your GPU memory and number. 

cp results/res18/CkptFile/1000.ckpt ../../model/detector.ckpt

# 6.Run classifier：

cd classifier and python adapt_ckpt.py --model1  net_detector_3 --model2  net_classifier_3  --resume ../detector/results/res18/CkptFile/1000.ckpt 

python main.py --model1  net_detector_3 --model2  net_classifier_3 -b 4 -b2 4 --save-dir net3 --resume ./results/start.ckpt --start-epoch 30 --epochs 130

python main.py --model1  net_detector_3 --model2  net_classifier_4 -b 4 -b2 4 --save-dir net4 --resume ./results/net3/130.ckpt --freeze_batchnorm 1 --start-epoch 121

cp results/net4/160.ckpt ../../model/classifier.ckpt

# Testing：

1.	unzip the stage 2 data 
2.	go to root folder
3.	open config_submit.py, filling in datapath with the stage 2 data path
4.	python main.py
5.	get the results from prediction.csv

if you have bug about short of memory, set the 'n_worker_preprocessing' in config\_submit.py to a int that is smaller than your core number.

# Brief Introduction to algorithm
Extra Data and labels: we use LUNA16 as extra data, and we manually labeled the locations of nodules in the stage1 training dataset. We also manually washed the label of LUNA16, deleting those that we think irrelavent to cancer. The labels are stored in ./training./detector./labels.

The training involves four steps
1. prepare data

    All data are resized to 1x1x1 mm, the luminance is clipped between -1200 and 600, scaled to 0-255 and converted to uint8. A mask that include the lungs is calculated, luminance of every pixel outside the mask is set to 170. The results will be stored in 'preprocess_result_path' defined in config_training.py along with their corresponding detection labels.

2. training a nodule detector

    in this part, a 3d faster-rcnn is used as the detector. The input size is 128 x 128 x 128, an online hard negative sample mining method is used. The network structure is based on U-net.
    
3. get all proposals
    
    The model trained in part 2 was tested on all data, giving all suspicious nodule locations and confidences (proposals)
    
4. training a cancer classifier
    
    For each case, 5 proposals are samples according to its confidence, and for each proposal a 96 x 96 x 96 cubes centered at the proposal center is cropped. 
    
    These proposals are fed to the detector and the feature in the last convolutional layer is extracted for each proposal. These features are fed to a fully-connected network and a cancer probability $P_i$ is calculated for each proposal. The cancer probability for this case is calculated as:

    $P = 1-(1-P_d)\Pi(1-P_i)$,
    
    where the $P_d$ stand for the probability of cancer of a dummy nodule, which is a trainable constant. It account for any possibility that the nodule is missed by the detector or this patient do not have a nodule now. Then the classification loss is calculated as the cross entropy between this $P$ and the label. 
    
    The second loss term is defined as: $-\log(P)\boldsymbol{1}(y_{nod}=1 \& P<0.03)$, which means that if this proposal is manually labeled as nodule and its probability is lower than 3%, this nodule would be forced to have higher cancer probability. Yet the effect of this term has not been carefully studied.
    
    To prevent overfitting, the network is alternatively trained on detection task and classification task.

The network archetecture is shown below

<img src="./images/nodulenet.png" width=50%>

<img src="./images/casenet.png" width=50%>
