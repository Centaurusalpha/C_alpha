from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pylab
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.layer2=nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3),nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.layer3=nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer4=nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )


        self.fc=nn.Sequential(
            nn.Linear(128*4*4,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,10) )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.layer5(x)
        x = x.view(x.size(0),-1)
        x=self.fc(x)
        return x





batch_size=64
learning_rate=1e-2
num_epoches=20
data_tf=transforms.Compose(
    [transforms.ToTensor(),transforms.Normalize([0.5],[0.5])]
)

train_dataset=datasets.MNIST(root='./data',train=True,transform=data_tf,
                             download=True)
test_dataset=datasets.MNIST(root='./data',train=False,transform=data_tf)
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
model=CNN().cuda()
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=learning_rate)
# model training


def train():
    epoch = 0
    for data in train_loader:
        img, label = data
        # img = img.view(img.size(0), -1)
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        out = model(img)
        loss = criterion(out, label)
        print_loss = loss.data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch += 1
        if epoch % 50 == 0:
            print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))


# model evaluation

def test():
    # model.eval()
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:
        img, label = data
        # img = img.view(img.size(0), -1)
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        # out = model(img)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.data.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
        eval_loss / (len(test_dataset)),
        eval_acc / (len(test_dataset))
    ))

#
# for _ in range(2):
#     train()
#     test()

x1=train_dataset.train_data[1]
print(x1)
plt.imshow(x1, cmap='gray')
pylab.show()




