from network import *
from function import *
from variate import *
import os
import sys

net = Net()
weightsInit(net,dev=0.01)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr = LR)
net.cuda()
net.train()

train_data = load_data(dataPath,dataCsvPath,shuffle=True)
ttest_data = load_data(ttestPath,ttestCsvPath,shuffle=False)
for epoch in range(EPOCH):
    num = 0
    for pic in train_data:
        
        img_data = pic['data']
        d_data = pic['csvData']

        num = num+1
        dMap = net(img_data,d_data)
        loss = net.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if num % 1000 == 0:
            print('epoch :',epoch,' ','photo :',num)

    path_name = os.path.join('./models/','{}.pkl'.format(epoch)) 
    torch.save(net.state_dict(),path_name)
    
    mae,mse  = evaluate(path_name,ttest_data)
    if mae < min_mae:
        min_mae = mae
        min_mse = mse
        min_epoch = epoch 
    print('current: ', epoch,' mae ',mae,' mse ',mse)
    print('best: ',min_epoch,' mae ',min_mae,' mse ',min_mse)

    

    


