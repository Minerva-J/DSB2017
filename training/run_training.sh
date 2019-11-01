#!/bin/bash
set -e

python prepare.py
cd detector
eps=1000
python main.py --model res18 -b 12 --epochs $eps --save-dir res18/CkptFile 
python main.py --model res18 -b 1 --resume results/res18/CkptFile/$eps.ckpt --test 1
cp results/res18/CkptFile/$eps.ckpt ../../model/detector.ckpt


cd ../classifier
python adapt_ckpt.py --model1  net_detector_3 --model2  net_classifier_3  --resume ../detector/results/res18/CkptFile/CkptFile.ckpt
python main.py --model1  net_detector_3 --model2  net_classifier_3 -b 4 -b2 4 --save-dir net3 --resume ./results/start.ckpt --start-epoch 30 --epochs 130
python main.py --model1  net_detector_3 --model2  net_classifier_4 -b 4 -b2 4 --save-dir net4 --resume ./results/net3/130.ckpt --freeze_batchnorm 1 --start-epoch 121
cp results/net4/160.ckpt ../../model/classifier.ckpt
