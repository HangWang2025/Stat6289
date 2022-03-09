#%%markdown
# # Question 1

#%%
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



batch_size = 4
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
           
#%%          
#for hiden layer is four           
import torch.nn as nn
import torch.nn.functional as F


from torch.nn.modules import dropout
class ReluModel(nn.Module):
  def __init__(self):
    super(ReluModel,self).__init__()
    self.fc = nn.Linear(32*32*3,512)
    self.fc0 = nn.Linear(512,256)
    self.fc1 = nn.Linear(256,128)
    self.fc2 = nn.Linear(128,64)
    self.fc3 = nn.Linear(128,10)
    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    # print(x.shape)
    x = torch.flatten(x, 1)
    # x = x.reshape(4,-1)
    # print(x.shape)
    # x = x.view()
    x = self.fc(x)
    # x = F.relu(self.fc(x))
    # print(x.shape)
    x = F.relu(self.fc0(x))
    # print(x.shape)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    # print(x.shape)
    x = F.relu(self.fc3(x))
    # print(x.shape)
    x = self.dropout(x)
    return x

rm = ReluModel()

#%%
#lost function

import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(rm.parameters(), lr=0.001, momentum=0.9)

acc_cnn=[]
for epoch in range(10):  # loop over the dataset multiple times
    train_accs=[]
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = rm(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        acc=(outputs.argmax(dim=-1)==labels).float().mean() #paint  The y-axis is the test (validation) accuracy and  the x-axis is the number of epochs
        train_accs.append(acc)                                                                
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    train_acc=sum(train_accs)/len(train_accs)
    acc_cnn.append(train_acc.cpu().item())
print('Finished Training')
#%%result
# [1,  2000] loss: 2.274
# [1,  4000] loss: 2.212
# [1,  6000] loss: 2.193
# [1,  8000] loss: 2.156
# [1, 10000] loss: 2.140
# [1, 12000] loss: 2.135
# [2,  2000] loss: 2.120
# [2,  4000] loss: 2.109
# [2,  6000] loss: 2.103
# [2,  8000] loss: 2.099
# [2, 10000] loss: 2.105
# [2, 12000] loss: 2.087
# [3,  2000] loss: 2.078
# [3,  4000] loss: 2.074
# [3,  6000] loss: 2.070
# [3,  8000] loss: 2.056
# [3, 10000] loss: 2.077
# [3, 12000] loss: 2.055
# [4,  2000] loss: 2.042
# [4,  4000] loss: 2.068
# [4,  6000] loss: 2.068
# [4,  8000] loss: 2.052
# [4, 10000] loss: 2.039
# [4, 12000] loss: 2.055
# [5,  2000] loss: 2.026
# [5,  4000] loss: 2.052
# [5,  6000] loss: 2.030
# [5,  8000] loss: 2.044
# [5, 10000] loss: 2.026
# [5, 12000] loss: 2.039
# [6,  2000] loss: 2.011
# [6,  4000] loss: 2.024
# [6,  6000] loss: 2.023
# [6,  8000] loss: 2.016
# [6, 10000] loss: 2.031
# [6, 12000] loss: 2.028
# [7,  2000] loss: 1.997
# [7,  4000] loss: 1.990
# [7,  6000] loss: 1.984
# [7,  8000] loss: 1.978
# [7, 10000] loss: 1.976
# [7, 12000] loss: 1.979
# [8,  2000] loss: 1.947
# [8,  4000] loss: 1.963
# [8,  6000] loss: 1.981
# [8,  8000] loss: 1.965
# [8, 10000] loss: 1.940
# [8, 12000] loss: 1.939
# [9,  2000] loss: 1.922
# [9,  4000] loss: 1.907
# [9,  6000] loss: 1.912
# [9,  8000] loss: 1.918
# [9, 10000] loss: 1.916
# [9, 12000] loss: 1.919
# [10,  2000] loss: 1.899
# [10,  4000] loss: 1.894
# [10,  6000] loss: 1.911
# [10,  8000] loss: 1.915
# [10, 10000] loss: 1.903
# [10, 12000] loss: 1.907
# # Finished Training
#%%

acc_cnn4 = acc_cnn
print(acc_cnn4)
#reasult
# [0.21240000426769257,
#  0.2553800046443939,
#  0.27017998695373535,
#  0.27526000142097473,
#  0.28200000524520874,
#  0.2872200012207031,
#  0.3078800141811371,
#  0.3227199912071228,
#  0.3443799912929535,
#  0.3462199866771698]

#%%
PATH = './cifar_net.pth'
torch.save(rm.state_dict(), PATH)
#%%
rm = ReluModel()
rm.load_state_dict(torch.load(PATH))
outputs = rm(images)
predicted = torch.max(outputs, 1)
#%%
# prepare to count predictions for each class
import matplotlib.pyplot as plt
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
lr=0.001
# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = rm(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
#             total_pred[classes[label]] += 1
# iters  = []
# train_acc = []
# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
#%%result
# Accuracy for class: plane is 56.5 %
# Accuracy for class: car   is 34.7 %
# Accuracy for class: bird  is 28.7 %
# Accuracy for class: cat   is 29.6 %
# Accuracy for class: deer  is 28.1 %
# Accuracy for class: dog   is 27.3 %
# Accuracy for class: frog  is 34.0 %
# Accuracy for class: horse is 31.3 %
# Accuracy for class: ship  is 34.5 %
# Accuracy for class: truck is 34.7 %
#%%markdown
# # THEN CONSIDER THE MODEL WITH TWO 0 hidden layer

#%%
import torch.nn as nn
import torch.nn.functional as F


from torch.nn.modules import dropout
class ReluModel(nn.Module):
  def __init__(self):
    super(ReluModel,self).__init__()
    self.fc = nn.Linear(32*32*3,512)
    self.fc0 = nn.Linear(512,10)
    # self.fc0 = nn.Linear(512,256)
    # self.fc1 = nn.Linear(256,128)
    # self.fc2 = nn.Linear(128,64)
    # self.fc3 = nn.Linear(128,10)
    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    # print(x.shape)
    x = torch.flatten(x, 1)
    # x = x.reshape(4,-1)
    # print(x.shape)
    # x = x.view()
    x = self.fc(x)
    # x = F.relu(self.fc(x))
    # print(x.shape)
    x = F.relu(self.fc0(x))
    # # print(x.shape)
    # x = F.relu(self.fc1(x))
    # # print(x.shape)
    # x = F.relu(self.fc3(x))
    # print(x.shape)
    x = self.dropout(x)
    return x

rm = ReluModel()
#%%
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(rm.parameters(), lr=0.001, momentum=0.9)
#%%
acc_cnn=[]
for epoch in range(10):  # loop over the dataset multiple times
    train_accs=[]
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = rm(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        acc=(outputs.argmax(dim=-1)==labels).float().mean() #paint  The y-axis is the test (validation) accuracy and  the x-axis is the number of epochs
        train_accs.append(acc) 
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    train_acc=sum(train_accs)/len(train_accs)
    acc_cnn.append(train_acc.cpu().item())
print('Finished Training')
#%%
# [1,  2000] loss: 2.231
# [1,  4000] loss: 2.183
# [1,  6000] loss: 2.176
# [1,  8000] loss: 2.167
# [1, 10000] loss: 2.163
# [1, 12000] loss: 2.161
# [2,  2000] loss: 2.156
# [2,  4000] loss: 2.158
# [2,  6000] loss: 2.160
# [2,  8000] loss: 2.151
# [2, 10000] loss: 2.136
# [2, 12000] loss: 2.160
# [3,  2000] loss: 2.137
# [3,  4000] loss: 2.146
# [3,  6000] loss: 2.153
# [3,  8000] loss: 2.153
# [3, 10000] loss: 2.157
# [3, 12000] loss: 2.151
# [4,  2000] loss: 2.140
# [4,  4000] loss: 2.139
# [4,  6000] loss: 2.161
# [4,  8000] loss: 2.157
# [4, 10000] loss: 2.145
# [4, 12000] loss: 2.138
# [5,  2000] loss: 2.137
# [5,  4000] loss: 2.139
# [5,  6000] loss: 2.154
# [5,  8000] loss: 2.148
# [5, 10000] loss: 2.150
# [5, 12000] loss: 2.135
# [6,  2000] loss: 2.141
# [6,  4000] loss: 2.149
# [6,  6000] loss: 2.145
# [6,  8000] loss: 2.157
# [6, 10000] loss: 2.149
# [6, 12000] loss: 2.148
# [7,  2000] loss: 2.137
# [7,  4000] loss: 2.132
# [7,  6000] loss: 2.141
# [7,  8000] loss: 2.147
# [7, 10000] loss: 2.151
# [7, 12000] loss: 2.152
# [8,  2000] loss: 2.137
# [8,  4000] loss: 2.122
# [8,  6000] loss: 2.142
# [8,  8000] loss: 2.138
# [8, 10000] loss: 2.140
# [8, 12000] loss: 2.143
# [9,  2000] loss: 2.135
# [9,  4000] loss: 2.139
# [9,  6000] loss: 2.142
# [9,  8000] loss: 2.149
# [9, 10000] loss: 2.129
# [9, 12000] loss: 2.129
# [10,  2000] loss: 2.123
# [10,  4000] loss: 2.137
# [10,  6000] loss: 2.144
# [10,  8000] loss: 2.153
# [10, 10000] loss: 2.135
# [10, 12000] loss: 2.151
# Finished Training
#%%
print(acc_cnn)
acc_cnn0=acc_cnn
# result
# [0.2330400049686432,
#  0.247639998793602,
#  0.2487799972295761,
#  0.2524600028991699,
#  0.2543199956417084,
#  0.24852000176906586,
#  0.25411999225616455,
#  0.25429999828338623,
#  0.2549799978733063,
#  0.2565999925136566]
#%%
PATH = './cifar_net.pth'
torch.save(rm.state_dict(), PATH)
outputs = rm(images)
predicted = torch.max(outputs, 1)

#%%
# prepare to count predictions for each class
import matplotlib.pyplot as plt
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
lr=0.001
# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = rm(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
#             total_pred[classes[label]] += 1
# iters  = []
# train_acc = []
# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
#%%



#%%markdown
# # THEN CONSIDER THE MODEL WITH TWO 1 hidden layer

#%%
import torch.nn as nn
import torch.nn.functional as F


from torch.nn.modules import dropout
class ReluModel(nn.Module):
  def __init__(self):
    super(ReluModel,self).__init__()
    self.fc = nn.Linear(32*32*3,512)
    self.fc0 = nn.Linear(512,256)
    self.fc1 = nn.Linear(256,10)
    # self.fc1 = nn.Linear(128,10)
    # self.fc2 = nn.Linear(128,64)
    # self.fc3 = nn.Linear(128,10)
    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    # print(x.shape)
    x = torch.flatten(x, 1)
    # x = x.reshape(4,-1)
    # print(x.shape)
    # x = x.view()
    x = self.fc(x)
    x = F.relu(self.fc0(x))
    # print(x.shape)
    x = F.relu(self.fc1(x))
    # # print(x.shape)
    
    # x = F.relu(self.fc2(x))
    # # print(x.shape)
    # x = F.relu(self.fc3(x))
    # print(x.shape)
    x = self.dropout(x)
    return x

rm = ReluModel()
#%%
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(rm.parameters(), lr=0.001, momentum=0.9)
#%%
acc_cnn=[]
for epoch in range(10):  # loop over the dataset multiple times
    train_accs=[]
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = rm(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        acc=(outputs.argmax(dim=-1)==labels).float().mean() #paint  The y-axis is the test (validation) accuracy and  the x-axis is the number of epochs
        train_accs.append(acc) 
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    train_acc=sum(train_accs)/len(train_accs)
    acc_cnn.append(train_acc.cpu().item())
print('Finished Training')
#%%
# [1,  2000] loss: 2.299
# [1,  4000] loss: 2.298
# [1,  6000] loss: 2.298
# [1,  8000] loss: 2.299
# [1, 10000] loss: 2.299
# [1, 12000] loss: 2.299
# [2,  2000] loss: 2.299
# [2,  4000] loss: 2.300
# [2,  6000] loss: 2.299
# [2,  8000] loss: 2.299
# [2, 10000] loss: 2.300
# [2, 12000] loss: 2.299
# [3,  2000] loss: 2.300
# [3,  4000] loss: 2.298
# [3,  6000] loss: 2.299
# [3,  8000] loss: 2.300
# [3, 10000] loss: 2.299
# [3, 12000] loss: 2.299
# [4,  2000] loss: 2.299
# [4,  4000] loss: 2.299
# [4,  6000] loss: 2.300
# [4,  8000] loss: 2.299
# [4, 10000] loss: 2.299
# [4, 12000] loss: 2.300
# [5,  2000] loss: 2.300
# [5,  4000] loss: 2.300
# [5,  6000] loss: 2.299
# [5,  8000] loss: 2.300
# [5, 10000] loss: 2.300
# [5, 12000] loss: 2.299
# [6,  2000] loss: 2.298
# [6,  4000] loss: 2.300
# [6,  6000] loss: 2.300
# [6,  8000] loss: 2.299
# [6, 10000] loss: 2.301
# [6, 12000] loss: 2.299
# [7,  2000] loss: 2.299
# [7,  4000] loss: 2.300
# [7,  6000] loss: 2.300
# [7,  8000] loss: 2.300
# [7, 10000] loss: 2.299
# [7, 12000] loss: 2.299
# [8,  2000] loss: 2.299
# [8,  4000] loss: 2.299
# [8,  6000] loss: 2.300
# [8,  8000] loss: 2.299
# [8, 10000] loss: 2.299
# [8, 12000] loss: 2.299
# [9,  2000] loss: 2.298
# [9,  4000] loss: 2.299
# [9,  6000] loss: 2.300
# [9,  8000] loss: 2.298
# [9, 10000] loss: 2.300
# [9, 12000] loss: 2.298
# [10,  2000] loss: 2.299
# [10,  4000] loss: 2.299
# [10,  6000] loss: 2.298
# [10,  8000] loss: 2.300
# [10, 10000] loss: 2.300
# [10, 12000] loss: 2.299
# Finished Training
#%%
print(acc_cnn)
acc_cnn1=acc_cnn
[0.12421999871730804, 0.11907999962568283, 0.12241999804973602, 0.12166000157594681, 0.12123999744653702, 0.12144000083208084, 0.12049999833106995, 0.12231999635696411, 0.12297999858856201, 0.12195999920368195]
#%%
PATH = './cifar_net.pth'
torch.save(rm.state_dict(), PATH)
outputs = rm(images)
predicted = torch.max(outputs, 1)

#%%
# prepare to count predictions for each class
import matplotlib.pyplot as plt
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
lr=0.001
# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = rm(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
#             total_pred[classes[label]] += 1
# iters  = []
# train_acc = []
# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
#%%

#%%markdown
# # THEN CONSIDER THE MODEL WITH TWO 2 hidden layer

#%%
import torch.nn as nn
import torch.nn.functional as F


from torch.nn.modules import dropout
class ReluModel(nn.Module):
  def __init__(self):
    super(ReluModel,self).__init__()
    self.fc = nn.Linear(32*32*3,512)
    self.fc0 = nn.Linear(512,256)
    self.fc1 = nn.Linear(256,128)
    # self.fc1 = nn.Linear(256,128)
    # self.fc2 = nn.Linear(128,64)
    self.fc2 = nn.Linear(128,10)
    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    # print(x.shape)
    x = torch.flatten(x, 1)
    # x = x.reshape(4,-1)
    # print(x.shape)
    # x = x.view()
    x = self.fc(x)
    # x = F.relu(self.fc(x))
    # print(x.shape)
    x = F.relu(self.fc0(x))
    # # print(x.shape)
    x = F.relu(self.fc1(x))
    # # print(x.shape)
    x = F.relu(self.fc2(x))
    # print(x.shape)
    x = self.dropout(x)
    return x

rm = ReluModel()
#%%
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(rm.parameters(), lr=0.001, momentum=0.9)
#%%
acc_cnn=[]
for epoch in range(10):  # loop over the dataset multiple times
    train_accs=[]
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = rm(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        acc=(outputs.argmax(dim=-1)==labels).float().mean() #paint  The y-axis is the test (validation) accuracy and  the x-axis is the number of epochs
        train_accs.append(acc) 
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    train_acc=sum(train_accs)/len(train_accs)
    acc_cnn.append(train_acc.cpu().item())
print('Finished Training')
# #%%
# [1,  2000] loss: 2.221
# [1,  4000] loss: 2.147
# [1,  6000] loss: 2.145
# [1,  8000] loss: 2.119
# [1, 10000] loss: 2.098
# [1, 12000] loss: 2.072
# [2,  2000] loss: 2.048
# [2,  4000] loss: 2.046
# [2,  6000] loss: 2.050
# [2,  8000] loss: 2.034
# [2, 10000] loss: 2.041
# [2, 12000] loss: 2.015
# [3,  2000] loss: 2.005
# [3,  4000] loss: 2.007
# [3,  6000] loss: 2.010
# [3,  8000] loss: 2.003
# [3, 10000] loss: 2.003
# [3, 12000] loss: 1.980
# [4,  2000] loss: 1.967
# [4,  4000] loss: 1.975
# [4,  6000] loss: 1.959
# [4,  8000] loss: 1.989
# [4, 10000] loss: 1.968
# [4, 12000] loss: 1.959
# [5,  2000] loss: 1.952
# [5,  4000] loss: 1.944
# [5,  6000] loss: 1.959
# [5,  8000] loss: 1.951
# [5, 10000] loss: 1.943
# [5, 12000] loss: 1.953
# [6,  2000] loss: 1.942
# [6,  4000] loss: 1.944
# [6,  6000] loss: 1.932
# [6,  8000] loss: 1.933
# [6, 10000] loss: 1.935
# [6, 12000] loss: 1.937
# [7,  2000] loss: 1.926
# [7,  4000] loss: 1.925
# [7,  6000] loss: 1.918
# [7,  8000] loss: 1.931
# [7, 10000] loss: 1.928
# [7, 12000] loss: 1.924
# [8,  2000] loss: 1.897
# [8,  4000] loss: 1.892
# [8,  6000] loss: 1.900
# [8,  8000] loss: 1.918
# [8, 10000] loss: 1.903
# [8, 12000] loss: 1.933
# [9,  2000] loss: 1.891
# [9,  4000] loss: 1.892
# [9,  6000] loss: 1.909
# [9,  8000] loss: 1.917
# [9, 10000] loss: 1.920
# [9, 12000] loss: 1.917
# [10,  2000] loss: 1.884
# [10,  4000] loss: 1.900
# [10,  6000] loss: 1.882
# [10,  8000] loss: 1.910
# [10, 10000] loss: 1.925
# [10, 12000] loss: 1.903
# Finished Training
#%%
acc_cnn2=acc_cnn
print(acc_cnn2)
# result
# [0.24804000556468964, 0.29660001397132874, 0.3130199909210205, 0.32486000657081604, 0.3327000141143799, 0.3388800024986267, 0.34345999360084534, 0.34940001368522644, 0.3478800058364868, 0.35221999883651733]
#%%
PATH = './cifar_net.pth'
torch.save(rm.state_dict(), PATH)
outputs = rm(images)
predicted = torch.max(outputs, 1)

#%%
# prepare to count predictions for each class
import matplotlib.pyplot as plt
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
lr=0.001
# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = rm(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
#             total_pred[classes[label]] += 1
# iters  = []
# train_acc = []
# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
#%%
#hiden layer3

import torch.nn as nn
import torch.nn.functional as F


from torch.nn.modules import dropout
class ReluModel(nn.Module):
  def __init__(self):
    super(ReluModel,self).__init__()
    self.fc = nn.Linear(32*32*3,512)
    self.fc0 = nn.Linear(512,256)
    self.fc1 = nn.Linear(256,128)
    # self.fc1 = nn.Linear(256,128)
    self.fc2 = nn.Linear(128,64)
    self.fc3 = nn.Linear(64,10)
    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    # print(x.shape)
    x = torch.flatten(x, 1)
    # x = x.reshape(4,-1)
    # print(x.shape)
    # x = x.view()
    x = self.fc(x)
    # x = F.relu(self.fc(x))
    # print(x.shape)
    x = F.relu(self.fc0(x))
    # # print(x.shape)
    x = F.relu(self.fc1(x))
    # # print(x.shape)
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    # print(x.shape)
    x = self.dropout(x)
    return x

rm = ReluModel()
#%%
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(rm.parameters(), lr=0.001, momentum=0.9)
#%%
acc_cnn=[]
for epoch in range(10):  # loop over the dataset multiple times
    train_accs=[]
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = rm(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        acc=(outputs.argmax(dim=-1)==labels).float().mean() #paint  The y-axis is the test (validation) accuracy and  the x-axis is the number of epochs
        train_accs.append(acc) 
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    train_acc=sum(train_accs)/len(train_accs)
    acc_cnn.append(train_acc.cpu().item())
print('Finished Training')
#%%
# [1,  2000] loss: 2.288
# [1,  4000] loss: 2.239
# [1,  6000] loss: 2.197
# [1,  8000] loss: 2.197
# [1, 10000] loss: 2.182
# [1, 12000] loss: 2.173
# [2,  2000] loss: 2.166
# [2,  4000] loss: 2.152
# [2,  6000] loss: 2.155
# [2,  8000] loss: 2.149
# [2, 10000] loss: 2.149
# [2, 12000] loss: 2.137
# [3,  2000] loss: 2.133
# [3,  4000] loss: 2.132
# [3,  6000] loss: 2.127
# [3,  8000] loss: 2.131
# [3, 10000] loss: 2.137
# [3, 12000] loss: 2.126
# [4,  2000] loss: 2.111
# [4,  4000] loss: 2.091
# [4,  6000] loss: 2.110
# [4,  8000] loss: 2.122
# [4, 10000] loss: 2.104
# [4, 12000] loss: 2.117
# [5,  2000] loss: 2.097
# [5,  4000] loss: 2.095
# [5,  6000] loss: 2.114
# [5,  8000] loss: 2.090
# [5, 10000] loss: 2.107
# [5, 12000] loss: 2.116
# [6,  2000] loss: 2.090
# [6,  4000] loss: 2.102
# [6,  6000] loss: 2.104
# [6,  8000] loss: 2.085
# [6, 10000] loss: 2.085
# [6, 12000] loss: 2.093
# [7,  2000] loss: 2.083
# [7,  4000] loss: 2.088
# [7,  6000] loss: 2.076
# [7,  8000] loss: 2.080
# [7, 10000] loss: 2.094
# [7, 12000] loss: 2.092
# [8,  2000] loss: 2.083
# [8,  4000] loss: 2.079
# [8,  6000] loss: 2.087
# [8,  8000] loss: 2.081
# [8, 10000] loss: 2.076
# [8, 12000] loss: 2.063
# [9,  2000] loss: 2.046
# [9,  4000] loss: 2.046
# [9,  6000] loss: 2.071
# [9,  8000] loss: 2.053
# [9, 10000] loss: 2.054
# [9, 12000] loss: 2.060
# [10,  2000] loss: 2.039
# [10,  4000] loss: 2.054
# [10,  6000] loss: 2.043
# [10,  8000] loss: 2.049
# [10, 10000] loss: 2.035
# [10, 12000] loss: 2.027
# Finished Training

#%%
acc_cnn3=acc_cnn
print(acc_cnn3)
#result
# [0.1677599996328354,
#  0.19884000718593597,
#  0.20764000713825226,
#  0.214819997549057,
#  0.21461999416351318,
#  0.21967999637126923,
#  0.22168000042438507,
#  0.22869999706745148,
#  0.2453400045633316,
#  0.2566399872303009]

#%%markdon

# # the original cnn

#%%
import torch
import torchvision
import torchvision.transforms as transforms
#%%
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#%%

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.ReLu(self.fc1(x))
        x = F.ReLu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
#%%
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#%%


acc_cnn=[]
for epoch in range(10):  # loop over the dataset multiple times
    train_accs=[]
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        acc=(outputs.argmax(dim=-1)==labels).float().mean() #paint  The y-axis is the test (validation) accuracy and  the x-axis is the number of epochs
        train_accs.append(acc) 
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    train_acc=sum(train_accs)/len(train_accs)
    acc_cnn.append(train_acc.cpu().item())
print('Finished Training')
#%%
# [1,  2000] loss: 2.304
# [1,  4000] loss: 2.305
# [1,  6000] loss: 2.306
# [1,  8000] loss: 2.304
# [1, 10000] loss: 2.304
# [1, 12000] loss: 2.304
# [2,  2000] loss: 2.304
# [2,  4000] loss: 2.304
# [2,  6000] loss: 2.305
# [2,  8000] loss: 2.306
# [2, 10000] loss: 2.304
# [2, 12000] loss: 2.304
# [3,  2000] loss: 2.305
# [3,  4000] loss: 2.305
# [3,  6000] loss: 2.305
# [3,  8000] loss: 2.304
# [3, 10000] loss: 2.302
# [3, 12000] loss: 2.306
# [4,  2000] loss: 2.305
# [4,  4000] loss: 2.305
# [4,  6000] loss: 2.304
# [4,  8000] loss: 2.304
# [4, 10000] loss: 2.305
# [4, 12000] loss: 2.304
# [5,  2000] loss: 2.304
# [5,  4000] loss: 2.304
# [5,  6000] loss: 2.305
# [5,  8000] loss: 2.305
# [5, 10000] loss: 2.304
# [5, 12000] loss: 2.305
# [6,  2000] loss: 2.304
# [6,  4000] loss: 2.306
# [6,  6000] loss: 2.305
# [6,  8000] loss: 2.303
# [6, 10000] loss: 2.305
# [6, 12000] loss: 2.305
# [7,  2000] loss: 2.304
# [7,  4000] loss: 2.303
# [7,  6000] loss: 2.305
# [7,  8000] loss: 2.305
# [7, 10000] loss: 2.305
# [7, 12000] loss: 2.305
# [8,  2000] loss: 2.304
# [8,  4000] loss: 2.304
# [8,  6000] loss: 2.304
# [8,  8000] loss: 2.306
# [8, 10000] loss: 2.305
# [8, 12000] loss: 2.304
# [9,  2000] loss: 2.304
# [9,  4000] loss: 2.305
# [9,  6000] loss: 2.304
# [9,  8000] loss: 2.304
# [9, 10000] loss: 2.305
# [9, 12000] loss: 2.305
# [10,  2000] loss: 2.304
# [10,  4000] loss: 2.305
# [10,  6000] loss: 2.305
# [10,  8000] loss: 2.303
#%%
acc_cnn_original=acc_cnn
print(acc_cnn_original)
#result
#[0.09991999715566635, 0.09991999715566635, 0.09991999715566635, 0.09991999715566635, 0.09991999715566635, 0.09991999715566635, 0.09991999715566635, 0.09991999715566635, 0.09991999715566635, 0.09991999715566635]


#%%
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

#%%
net = Net()
net.load_state_dict(torch.load(PATH))

#%%
outputs = net(images)
#%%
predicted = torch.max(outputs, 1)
#%%

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

#%%
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

#%%


#%%markdown

# # figure step

#%%
import numpy as np
import matplotlib.pyplot as plt
epochlist = np.array([1,2,3,4,5,6,7,8,9,10])
plt.plot(epochlist,acc_cnn0,linewidth = 2,label = r'$hiden layer=0$')
plt.plot(epochlist,acc_cnn1,linewidth = 2,label = r'$hiden layer=1$')
plt.plot(epochlist,acc_cnn2,linewidth = 2,label = r'$hiden layer=2$')
plt.plot(epochlist,acc_cnn3,linewidth = 2,label = r'$hiden layer=3$')
plt.plot(epochlist,acc_cnn4,linewidth = 2,label = r'$hiden layer=4$')
plt.plot(epochlist,acc_cnn_original,linewidth = 2,label = r'$CNN$')

plt.legend(loc = 'lower right')

plt.show()


#%%markdown
# # Question 2


#%%sigmoid cnn

#%%
import torch
import torchvision
import torchvision.transforms as transforms
#%%
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#%%

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.sigmoid(self.fc1(x))
        x = F.sigmid(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
#%%
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#%%


acc_cnn=[]
for epoch in range(10):  # loop over the dataset multiple times
    train_accs=[]
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        acc=(outputs.argmax(dim=-1)==labels).float().mean() #paint  The y-axis is the test (validation) accuracy and  the x-axis is the number of epochs
        train_accs.append(acc) 
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    train_acc=sum(train_accs)/len(train_accs)
    acc_cnn.append(train_acc.cpu().item())
print('Finished Training')

#%%
# [1,  2000] loss: 2.316
# [1,  4000] loss: 2.314
# [1,  6000] loss: 2.312
# [1,  8000] loss: 2.306
# [1, 10000] loss: 2.243
# [1, 12000] loss: 2.072
# [2,  2000] loss: 1.976
# [2,  4000] loss: 1.910
# [2,  6000] loss: 1.835
# [2,  8000] loss: 1.754
# [2, 10000] loss: 1.718
# [2, 12000] loss: 1.657
# [3,  2000] loss: 1.619
# [3,  4000] loss: 1.592
# [3,  6000] loss: 1.535
# [3,  8000] loss: 1.506
# [3, 10000] loss: 1.492
# [3, 12000] loss: 1.466
# [4,  2000] loss: 1.431
# [4,  4000] loss: 1.416
# [4,  6000] loss: 1.389
# [4,  8000] loss: 1.375
# [4, 10000] loss: 1.371
# [4, 12000] loss: 1.359
# [5,  2000] loss: 1.309
# [5,  4000] loss: 1.303
# [5,  6000] loss: 1.295
# [5,  8000] loss: 1.265
# [5, 10000] loss: 1.267
# [5, 12000] loss: 1.254
# [6,  2000] loss: 1.231
# [6,  4000] loss: 1.204
# [6,  6000] loss: 1.195
# [6,  8000] loss: 1.185
# [6, 10000] loss: 1.211
# [6, 12000] loss: 1.200
# [7,  2000] loss: 1.149
# [7,  4000] loss: 1.152
# [7,  6000] loss: 1.131
# [7,  8000] loss: 1.145
# [7, 10000] loss: 1.127
# [7, 12000] loss: 1.138
# [8,  2000] loss: 1.081
# [8,  4000] loss: 1.082
# [8,  6000] loss: 1.069
# [8,  8000] loss: 1.117
# [8, 10000] loss: 1.096
# [8, 12000] loss: 1.098
# [9,  2000] loss: 1.042
# [9,  4000] loss: 1.036
# [9,  6000] loss: 1.036
# [9,  8000] loss: 1.052
# [9, 10000] loss: 1.043
# [9, 12000] loss: 1.046
# [10,  2000] loss: 0.975
# [10,  4000] loss: 1.007
# [10,  6000] loss: 1.000
# [10,  8000] loss: 1.017
# [10, 10000] loss: 1.028
# [10, 12000] loss: 1.011
# Finished Training
#%%
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

#%%
net = Net()
net.load_state_dict(torch.load(PATH))

#%%
outputs = net(images)
#%%
predicted = torch.max(outputs, 1)
#%%

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

#%%
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

#%%markdown


# # Question3

#%%markdown
# # The code of q3 is almost same as above, just change the part of dropout part and the transform part 
# #
#%%
#%%
import torch
import torchvision
import torchvision.transforms as transforms
#%%
transform = transforms.Compose(
    [transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#%%

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.Relu(self.fc1(x))
        x = F.Relu(self.fc2(x))
        x = self.fc3(x)
        x = self.dropout(x)
        
        return x


net = Net()
#%%
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#%%


acc_cnn=[]
for epoch in range(10):  # loop over the dataset multiple times
    train_accs=[]
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        acc=(outputs.argmax(dim=-1)==labels).float().mean() #paint  The y-axis is the test (validation) accuracy and  the x-axis is the number of epochs
        train_accs.append(acc) 
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    train_acc=sum(train_accs)/len(train_accs)
    acc_cnn.append(train_acc.cpu().item())
print('Finished Training')

#%%
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

#%%
net = Net()
net.load_state_dict(torch.load(PATH))

#%%
outputs = net(images)
#%%
predicted = torch.max(outputs, 1)
#%%

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

#%%
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
#%%
# transform = transforms.Compose(
#  [transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
# transforms.ToTensor(),
# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],)    