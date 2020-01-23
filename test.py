import torch
import numpy as np
from  variate import *
from function import *
import random
from network import *
from PIL import Image
from pyheatmap.heatmap import HeatMap
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import *
import sys
import cv2
import matplotlib.pyplot as plt
import copy
import json
name = ""
class mainwindow(QMainWindow):
    def __init__(self):
        super().__init__()
        #self.resize(1000,600)
        self.initUI()
    def initUI(self):
    
        testAct = QAction('&打开图片',self)
        testAct.triggered.connect(self.test)
        testAct0 = QAction('&测试',self)
        testAct0.triggered.connect(self.test0)

        testAct1 = QAction('&批测试',self)
        testAct1.triggered.connect(self.test1)

 
        menubar = self.menuBar()
        testMenu = menubar.addMenu("&测试")
        testMenu.addAction(testAct)
        testMenu.addAction(testAct0)

        testMenu1 = menubar.addMenu("&批测试")
        testMenu1.addAction(testAct1)

        self.label = QLabel(self)
        self.label.setFixedSize(700, 800)
        self.label.move(50, 100)
        self.label.setStyleSheet("QLabel{background:white;}")

        self.label1 = QLabel(self)
        self.label1.setFixedSize(700, 500)
        self.label1.move(850, 100)
        self.label1.setStyleSheet("QLabel{background:white;}")

        self.label2 = QLabel(self)
        self.label2.setFixedSize(700, 250)
        self.label2.move(850, 650)
        self.label2.setStyleSheet("QLabel{background:white;}")

        self.setGeometry(100,100,1600,1000)
        self.setWindowTitle("大作业")
        self.show()
    def test1(self):
        net = Net()
        net.load_state_dict(torch.load(model_path))

        net.cuda()
        net.eval()

        data = load_data(test_path,test_csv_path,shuffle=False)
        mae = 0.0
        mse = 0.0
        for pic in data:
            img_data = pic['data']
            d_data = pic['csvData']
            dMap = net(img_data,d_data)
            dMap = dMap.cpu().data.numpy()
            count1 = np.sum(d_data)
            count2 = np.sum(dMap)
            print("truth: ",count1)
            print("predict: ",count2)

            mae +=abs(count1 - count2)
            mse +=((count1 - count2)*(count1 - count2))


        mae = mae/data.getNum()
        mse = np.sqrt(mse/data.getNum())
        print ('\nMAE: %f, MSE: %f' % (mae,mse))
    def test0(self):
        net = Net()
        net.load_state_dict(torch.load(model_path))

        net.cuda()
        net.eval()
        mae = 0.0
        mse = 0.0 

        global name

        img = cv2.imread(name,0)
        img = img.astype(np.float32, copy=False)
        ht = int((img.shape[0]/4)*4)
        wd = int((img.shape[1]/4)*4)
        img = cv2.resize(img,(wd,ht))
        img = img.reshape((1,1,img.shape[0],img.shape[1]))

        for index, ch in enumerate(name):
            if ch =="/":
                num = index
        csvName = './data/test_csv/'+name[num+1:-4]+'.csv'
        den = pd.read_csv((csvName), sep=',',header=None).values
        den  = den.astype(np.float32, copy=False)

        wd_1 = int(wd/4)
        ht_1 = int(ht/4)
        den = cv2.resize(den,(wd_1,ht_1))
        den = den * ((wd*ht)/(wd_1*ht_1)) 
        den = den.reshape((1,1,den.shape[0],den.shape[1])) 
        
      
        dMap = net(img,den)
        dMap = dMap.cpu().data.numpy()

        count1 = np.sum(den)
        count2 = np.sum(dMap)
     
        print("truth: ",count1)
        print("predict: ",count2)
        mae +=abs(count1 - count2)
        mse +=((count1 - count2)*(count1 - count2))
        mae = mae
        mse = np.sqrt(mse)
        print('mae: ',mae)
        print('mse: ',mse)

        t = np.max(dMap)
        density_map = 255*dMap/t 
        img1= density_map[0][0]

        h1 = int(img1.shape[0]*2)
        w1 = int(img1.shape[1]*2)
        h2 = img1.shape[0]
        w2 = img1.shape[1]

        newimg = np.zeros((h1,w1),dtype = np.uint8)
        for i in range(h1):
            for j in range(w1):
                hh = int(i/2)
                ww = int(j/2)
                u = i/2 - hh
                v = j/2 -ww
                sh = min(hh+1,h2-1)
                sw = min(w2-1,ww+1)
                newimg[i,j] = (1-u)*(1-v)*img1[hh,ww]+u*v*img1[sh,sw]+(1-u)*v*img1[hh,sw]+(1-v)*u*img1[sh,ww]

        cv2.imwrite('./temp/densityMap.bmp',newimg)
        
        self.label1.setPixmap(QPixmap('./temp/densityMap.bmp'))

        font = QtGui.QFont()
        font.setFamily("Times New Roman") 
        font.setPointSize(15)  
        self.label2.setFont(font)

        self.label2.setText("Truth : "+str(count1)+"\n\n"+"Predict : "+str(count2))
      
    def test(self):
        imgName,imgType = QFileDialog.getOpenFileName(self,"","")
        global name
        if imgName:
            for index, ch in enumerate(imgName):
                if ch =="/":
                    num = index
            name = './data/test/'+imgName[num+1:]
        img = cv2.imread(name)
        
        h1 = int(img.shape[0]/2)
        w1 = int(img.shape[1]/2)
        h2 = img.shape[0]
        w2 = img.shape[1]

        newimg = np.zeros((h1,w1,3),dtype = np.uint8)
        for i in range(h1):
            for j in range(w1):
                hh = int(i*2)
                ww = int(j*2)
                u = i*2 - hh
                v = j*2 -ww
                sh = min(hh+1,h2-1)
                sw = min(w2-1,ww+1)
                newimg[i,j] = (1-u)*(1-v)*img[hh,ww]+u*v*img[sh,sw]+(1-u)*v*img[hh,sw]+(1-v)*u*img[sh,ww]
        cv2.imwrite('./temp/temp.bmp',newimg)    
        self.label.setPixmap(QPixmap('./temp/temp.bmp'))

if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    ex = mainwindow()
    sys.exit(app.exec_())