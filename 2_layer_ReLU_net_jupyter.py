
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.linear_model import LinearRegression


device = torch.device("cuda")


# In[31]:


# parameters
h1 = 5
h2 = 100
W = np.array([1 if np.random.rand() > 0.5 else -1 for _ in range(h2)])
W = torch.from_numpy(W).float().to(device)
n_epoch = 1000
lr = 0.0002


# In[79]:


def calcH_min_eig(X):
    n = len(X)
    H = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            if i == j:
                H[i][j] = 0.5
            else:
                H[i][j] = X[i].dot(X[j]) * (0.5 - np.arccos(X[i].dot(X[j])) / (2 * np.pi))
    return H, np.linalg.eig(H), min(np.linalg.eig(H)[0])


# In[37]:


# data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)

ind = []
N = 50
n = 0
while n < N:
    index = np.random.randint(60000)
    if trainset.train_labels[index].item() == 0:
        n += 1
        ind.append(index)
n = 0
while n < N:
    index = np.random.randint(60000)
    if trainset.train_labels[index].item() == 1:
        n +=1
        ind.append(index)
print('samples:', ind)

trainset.train_data = trainset.train_data[ind].float()
trainset.train_labels = trainset.train_labels[ind]
trainset.train_data = trainset.train_data.view(-1, 28*28)

for i in range(2*N):
    trainset.train_data[i] = trainset.train_data[i] / float(np.linalg.norm(trainset.train_data[i]))

H, H_eig, lambda0 = calcH_min_eig(trainset.train_data.numpy())
print('\lambda_0:', lambda0)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=1)


# In[38]:


# 2-layer network
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(28*28, h2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = W.dot(x)
        return x


# 3-layer network
class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.fc1 = nn.Linear(28*28, h1)
        self.fc2 = nn.Linear(h1, h2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = W.dot(x)
        return x


# In[39]:


net = Net2().to(device)
criterion = nn.MSELoss()


# train
Loss_record = []
optimizer = optim.SGD(net.parameters(), lr=lr)

for epoch in range(n_epoch):
    running_loss = 0.0
    for i, data in enumerate(zip(trainset.train_data, trainset.train_labels)):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.float().to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if (epoch + 1) % 50 == 0:
        print('%d loss: %.8f' % (epoch + 1, running_loss))
    Loss_record.append(running_loss)

print('Finished Training')


# In[91]:


x_axis = np.array([lr * i for i in range(len(Loss_record))])
plt.scatter(x_axis, [l ** 2 for l in Loss_record], marker='.')
plt.scatter(x_axis, Loss_record[0] ** 2 * np.exp(- 2000 * lambda0 * x_axis), marker='.')
plt.scatter(x_axis, [Loss_record[0] ** 2 * (1 - 2000 * lr * lambda0 / 2) ** k for k, l in enumerate(Loss_record)], marker='.')
plt.show()


# In[81]:


lambda0


# In[89]:


def fit_rate(y):
    # y^2 = y[0]^2 * exp(-t*x), want t
    return sum([np.log(y[k-1] / y[k]) * 2 / lr for k in range(1, len(y))]) / (len(y)-1)
print(fit_rate(Loss_record), lambda0)
print(np.exp(-fit_rate(Loss_record)), np.exp(-lambda0))


# In[84]:


def fit_rate_discrete(y):
    # y[k]^2 = y[0]^2 * t^k, want t
    return sum([(y[k] ** 2 / y[k-1] ** 2) for k in range(1, len(y))]) / (len(y)-1)
print(fit_rate_discrete(Loss_record), 1-lr*lambda0/2)
print(1-fit_rate_discrete(Loss_record), lr*lambda0/2)


# In[117]:


v = np.random.normal(size=2*N)
-2*v.T.dot(H).dot(v), -lambda0 * np.linalg.norm(v)**2


# In[125]:


sorted(np.linalg.eig(H[:50,:50])[0]),sorted(np.linalg.eig(H)[0])


# In[128]:


print(np.linalg.eig(np.array([[0.5,0.3],[0.3,0.4]]))[0], 
np.linalg.eig(np.array([[0.5,0.5,0.3,0.3],[0.5,0.5,0.3,0.3],[0.3,0.3,0.4,0.4],[0.3,0.3,0.4,0.4]]))[0])

