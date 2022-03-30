#%%
import torch
import torchvision
import torchvision.transforms as transforms
#%%
transform = transforms.Compose(
    [transforms.ToTensor(),
    #  transforms.Lambda(lambda x: x.repeat(1,1,1)),
     transforms.Normalize((0.5), (0.5))])

batch_size =  32

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False,download=True, transform=transform)  

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,drop_last=True,shuffle=True, num_workers=2)

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
#%%
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
#%%
from torch._C import device
class RNN_Model(nn.Module):
  def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
    super(RNN_Model, self).__init__()
    self.hidden_dim = hidden_dim
    # print('bbbbbbb')
    self.layer_dim = layer_dim
    # print('aaaaaaa')
    self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, nonlinearity='relu')
    # print('ccccccc')
    self.fc = nn.Linear(hidden_dim, output_dim)
  
  def forward(self, x):
    h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
    print('aaaaa',x.shape)
    out, hn = self.rnn(x, h0.detach())  # avoid to 梯度爆炸
    print('bbbbb',x.shape)
    out = self.fc(out[:, -1, :]) 
    print(out.shape)
    return out
#%%
input_dim = 28
hidden_dim = 100
layer_dim = 2
output_dim = 10

model = RNN_Model(input_dim, hidden_dim, layer_dim, output_dim)

device = torch.device('cuda:0' if torch.cuda.is_available() else'cpu')
#%%
sequence_dim = 28
acc_rnn = []
loss_list = []
iteration_list = []

iter = 0
for epoch in range(10):
  for i, (images,labels) in enumerate(trainloader):
    optimizer.zero_grad()
    model.train=()
    images = images.view(-1, sequence_dim, input_dim)#.requires_grad_().to(device)
    print(images.shape,'bbbbbb')
    labels = labels.to(device)
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    acc=(outputs.argmax(dim=-1)==labels).float().mean() #paint  The y-axis is the test (validation) accuracy and  the x-axis is the number of epochs
    train_accs.append(acc)                                                                
        # print statistics
    running_loss += loss.item()
    if i % 200 == 199:    # print every 2000 mini-batches
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
        running_loss = 0.0
    train_acc=sum(train_accs)/len(train_accs)
    acc_cnn.append(train_acc.cpu().item())
print('Finished Training')
#%%