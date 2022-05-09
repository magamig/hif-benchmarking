# MoG-DCN
This repository contains the Pytorch codes of paper "Model-Guided Deep Hyperspectral Image Super-resolution"   

**Using sf=8 and trined /tested on CAVE as an example ,I will introduce the usage of this code**  
Download [CAVE](https://www.cs.columbia.edu/CAVE/databases/multispectral/) ,  [Harvard](http://vision.seas.harvard.edu/hyperspec/d2x5g3/) and [WV2](https://www.harrisgeospatial.com/Data-Imagery/https://www.harrisgeospatial.com/Data-Imagery/)
## Prepare the training data and the test data   
   1) Divide the dataset for training and testing respectively
   2) put training data and testing data in .mat format in corresponding folders
   3) run creat_pathlist.py to create *.txt ,for example './pathlist/datalist_NSSR_P.txt'
   4) By the way , you can change the Data reading method by changing ./*/clean_dataset.py
## train 
   Run ./sf_8_CAVE/train.py
## test    
   Run ./sf_8_CAVE/tst.py
## Contact  
Weisheng Dong, Email: wsdong@mail.xidian.edu.cn  
Chen Zhou, Email: zhouchen_7@163.com  

