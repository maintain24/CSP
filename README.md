# Structure Prediction

In this project, our team applied different machine learning algorithms with the goal of predicting the crystal structure of test data consisting of databases such as Crystallography Open Database (COD) and The Inorganic Crystal Structure Database (ICSD). The model used semantic segmentation methods, focusing on strengthening the model's learning of 3D structural feature information. In addition, we also tried long short-term memory (LSTM) models and automated machine learning methods. This model can assist in the discovery of unknown crystal structures to a certain extent.


##  Data Preparation
### Merge cif data and convert it into .npy file

python data_preprocessing.py

## Enter the specified folder CSP

cd /mnt/pycharm_project_CSP

## train 
### Run the training script

>sh tool/train.sh s3dis CSP_repro

###  Hang in the background train and output the log file train.log to the specified folder

>nohup sh tool/train.sh s3dis CSP_repro > /mnt/pycharm_project_CSP/dataset/run.log 2>&1 &

### Or choose to monitor the log content in the terminal

>tail -f /mnt/pycharm_project_CSP/dataset/run.log

## Test
### Specify GPU debugging, such as GPU 0
>CUDA_VISIBLE_DEVICES=0 sh tool/test.sh s3dis CSP_repro

### Hang in the background test and output the log file test.log to the specified folder
>nohup sh tool/test.sh s3dis CSP_repro > /mnt/pycharm_project_CSP/dataset/test.log 2>&1 &

### Or choose to monitor the log content in the terminal
>tail -f /mnt/pycharm_project_PT/dataset/test.log
