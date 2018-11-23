from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class Batch_Net(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,n_hidden_3,out_dim):
        super(Batch_Net, self).__init__()
        self.layer1=nn.Sequential(nn.Linear(in_dim,n_hidden_1),
            nn.BatchNorm1d(n_hidden_1),nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
            nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                nn.BatchNorm1d(n_hidden_3), nn.ReLU(True))
        self.layer4=nn.Sequential(nn.Linear(n_hidden_3,out_dim))

    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
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
model=Batch_Net(28*28,300,300,200,10).cuda()
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=learning_rate)
# model training

def train():
    epoch = 0
    for data in train_loader:
        img, label = data
        img = img.view(img.size(0), -1)
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
    model.eval()
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:
        img, label = data
        img = img.view(img.size(0), -1)
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
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

for _ in range(5):
    train()
    test()


