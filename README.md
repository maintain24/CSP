# Structure Prediction

In this project, our team applied different machine learning algorithms with the goal of predicting the crystal structure of test data consisting of databases such as Crystallography Open Database (COD) and The Inorganic Crystal Structure Database (ICSD). The model used semantic segmentation methods, focusing on strengthening the model's learning of 3D structural feature information. In addition, we also tried long short-term memory (LSTM) models and automated machine learning methods. This model can assist in the discovery of unknown crystal structures to a certain extent.


##  Data Preparation

StackEdit stores your files in your browser, which means all your files are automatically saved locally and are accessible **offline!**

## 进入指定文件夹CSP

cd /mnt/pycharm_project_CSP

## train 
### tool文件夹下的train.sh脚本

>sh tool/train.sh s3dis CSP_repro

###  挂后台训练并输出日志文件run.log到指定文件夹

>nohup sh tool/train.sh s3dis CSP_repro > /mnt/pycharm_project_CSP/dataset/run.log 2>&1 &

### 或者选择在终端中监控日志内容：

>tail -f /mnt/pycharm_project_CSP/dataset/run.log

## Test
### 可指定GPU，例如第0台GPU
>CUDA_VISIBLE_DEVICES=0 sh tool/test.sh s3dis CSP_repro

### 挂后台测试并输出日志文件test.log到指定文件夹
>nohup sh tool/test.sh s3dis CSP_repro > /mnt/pycharm_project_CSP/dataset/test.log 2>&1 &

### 或者直接在终端监控日志内容
>tail -f /mnt/pycharm_project_PT/dataset/test.log
