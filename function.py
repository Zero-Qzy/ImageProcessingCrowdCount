import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
import os
import random
import pandas as pd


#----------------------------数据转换--------------------------

def np2value(data,is_train = False,dtype = torch.FloatTensor):
    if is_train:
        v = Variable(torch.from_numpy(data).type(dtype))
    else:
        with torch.no_grad():
            v = Variable(torch.from_numpy(data).type(dtype))
    v = v.cuda()
    return v
#----------------------------模型初始化--------------------------
def weightsInit(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weightsInit(m, dev)
    else:
        for m in model.modules():#子层而非当前层,正态分布处理
            if isinstance(m, nn.Conv2d):#处理卷积核和偏置值                
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):#处理隐藏层
                m.weight.data.normal_(0.0, dev)

#----------------------------数据加载--------------------------

class load_data():
    def __init__(self, dataPath, CsvPath, shuffle=False):
        
        self.dataPath = dataPath
        self.CsvPath = CsvPath
        self.data_files = [filename for filename in os.listdir(dataPath) \
                           if os.path.isfile(os.path.join(dataPath,filename))]
        self.data_files.sort()
        self.shuffle = shuffle
        if shuffle:
            random.seed(2019)
        self.numSamples = len(self.data_files)
        self.blob_list = {}        
        self.id_list = range(0,self.numSamples)
    
        print ('Pre-loading the data.')
        idx = 0
        for fname in self.data_files:
            img = cv2.imread(os.path.join(self.dataPath,fname),0)
            img = img.astype(np.float32, copy=False)
            ht = img.shape[0]
            wd = img.shape[1]
            ht_1 = int((ht/4)*4)
            wd_1 = int((wd/4)*4)
           
            img = cv2.resize(img,(wd_1,ht_1))
            img = img.reshape((1,1,img.shape[0],img.shape[1]))
        
            den = pd.read_csv(os.path.join(self.CsvPath,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).values        
            den  = den.astype(np.float32, copy=False)
               
            wd_1 = int(wd_1/4)
            ht_1 = int(ht_1/4)
            den = cv2.resize(den,(wd_1,ht_1))                
            den = den * ((wd*ht)/(wd_1*ht_1)) 
            den = den.reshape((1,1,den.shape[0],den.shape[1]))     
            
                   
            blob = {}
            blob['data']=img
            blob['csvData']=den
           
            self.blob_list[idx] = blob
            idx = idx+1
            if idx % 500 == 0:                    
                print ('Loaded ', idx, '/', self.numSamples, 'files')
               
        print ('Completed Loading ', idx, 'files')

    def __iter__(self):
        if self.shuffle:                     
            random.shuffle(list(self.id_list))        
        files = self.data_files
        id_list = self.id_list
        for idx in id_list:
            blob = self.blob_list[idx]    
            blob['idx'] = idx 
            yield blob
    def getNum(self):
        return self.numSamples
                






        
            
        
