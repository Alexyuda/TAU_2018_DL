import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


use_cuda = torch.cuda.is_available()
dataDir = "D:/TAU/EX1/EX1/Q2"
input_csv_dir = dataDir + "/iris.data.csv"

#hyperparameters
batch_size = 5
lr = 0.0002
num_epoch = 500


def parse_iris_input(data_dir):
    # read and pharse input data
    data = pd.read_csv(data_dir, names=['f1', 'f2', 'f3', 'f4', 'species'])
    # change string value to numeric
    data.loc[data['species'] == 'Iris-setosa', 'species'] = 0
    data.loc[data['species'] == 'Iris-versicolor', 'species'] = 1
    data.loc[data['species'] == 'Iris-virginica', 'species'] = 2
    data = data.apply(pd.to_numeric)
    return data
    f.close()


class IrisNetDataSet(Dataset):
    def __init__(self, x, y):
        self.len = x.shape[0]
        self.x_data = torch.Tensor(x).float()
        self.y_data = torch.Tensor(y).long()

    def __getitem__(self, index):
        return self.x_data[index,:], self.y_data[index]

    def __len__(self):
        return len(self.x_data)

class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.requ(x)
        x = self.fc2(x)
        return x

    @staticmethod
    def requ(x):
        x = F.relu(x)
        x_2 = torch.mul(x, x)
        return x_2


def main():
    # read data and build
    data = parse_iris_input(input_csv_dir)
    # split to train and test
    X_train, X_test, y_train, y_test = train_test_split(data[data.columns[0:4]].values,
                                                        data.species.values, test_size=0.3)


    # split test to test and validation
    X_test, X_val, y_test, y_val = train_test_split(X_test,
                                                    y_test, test_size=0.5)
    X_test = Variable(torch.Tensor(X_test).float())
    X_val = Variable(torch.Tensor(X_val).float())
    y_test = Variable(torch.Tensor(y_test).long())
    y_val = Variable(torch.Tensor(y_val).long())

    if use_cuda:
        X_test = X_test.cuda()
        X_val = X_val.cuda()
        y_test = y_test.cuda()
        y_val = y_val.cuda()

    # build data loader
    dataSetTrain = IrisNetDataSet(X_train,y_train)
    train_loader = DataLoader(dataset=dataSetTrain, batch_size=batch_size, shuffle=True)

    # build net
    net = IrisNet()
    if use_cuda:
        net = net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr)

    # train
    train_losses = []
    validation_losses=[]
    fig = plt.figure()
    for ep_n,epoch in enumerate(range(num_epoch)):
        epoch_train_loss = 0
        epoch_counter = 0
        for data in train_loader:
            optimizer.zero_grad()
            input, target = data
            input, target = Variable(input), Variable(target)
            if use_cuda:
                input = input.cuda()
                target = target.cuda()
            out = net(input)
            loss = criterion(out,target)
            epoch_counter += 1
            epoch_train_loss += loss
            loss.backward()
            optimizer.step()
        epoch_train_loss = epoch_train_loss.data.cpu().numpy()/epoch_counter
        train_losses.append(epoch_train_loss)
        # evaluate on validation set
        loss_validation = criterion(net(X_val),y_val)
        validation_losses.append(loss_validation.data.cpu().numpy())
        print('ephoch: %d loss: %f' %(ep_n,loss_validation.data.cpu().numpy()))

        plt.plot(train_losses,color= 'b')
        plt.plot(validation_losses, color='r')
        plt.xlabel('epoch', fontsize=16)
        plt.ylabel('loss', fontsize=16)
        plt.title('Training on Iris Data set')
        plt.legend(['training loss','validation loss'])
        plt.pause(0.01)

    # get prediction
    out = net(X_test)
    _, predicted = torch.max(out.data.cpu(), 1)
    # get accuration
    test_loss = criterion(out, y_test)
    print('test lost %f ' % test_loss)
    print('Accuracy of the network %d %%' % (100 * torch.sum(y_test.data.cpu() == predicted) / len(y_test)))
    fig.savefig('Q2_results.png')



main()
