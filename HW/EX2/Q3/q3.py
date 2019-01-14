import torch
import torchfile
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction import image
import matplotlib.image as mpimg
from torch.utils.data import Dataset,DataLoader
from torchsummary import summary
import torch.optim as optim
import cv2

class aflwDataSet(Dataset):
    def __init__(self, x, y):
        self.len = len(y)
        self.x_data = x
        self.y_data = y

    def __getitem__(self, index):
        return torch.Tensor(self.x_data[index]).float(), torch.Tensor(self.y_data[index]).long()

    def __len__(self):
        return len(self.y_data)

class View(torch.nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

def pascal_negative_mining(net,personDir,pascalDir,net_input_dim,patch_dim, max_patches,use_cuda):
    patches = []

    # Create a list of images without a person
    #(in person_train.txt a line with label == -1 is a negative sample)
    myarray = np.loadtxt(personDir)
    myarray_new = np.zeros(myarray.shape[0])
    j = 0
    for i,line in enumerate(myarray):
        if (line[1] == -1):
            myarray_new[j] = line[0]
            j += 1
    neg_file_array  = myarray_new[:j]

    # create patches
    i = 0
    for line in neg_file_array:
        filename = str(int(line)).zfill(6) + '.jpg'
        img = mpimg.imread(pascalDir + filename)/255
        for i in range(0,img.shape[0]- patch_dim - 1,patch_dim):
            for j in range(0,img.shape[1]- patch_dim - 1,patch_dim):
                patch = img[i:i+patch_dim,j:j+patch_dim,:]

                patch_resized = cv2.resize(patch, (net_input_dim, net_input_dim))
                patch_resized = np.moveaxis(patch_resized, -1, 0)
                patch = np.moveaxis(patch, -1, 0)
                with torch.set_grad_enabled(False):
                    if use_cuda:
                        res = net(torch.Tensor(patch_resized).float().cuda().unsqueeze_(0))
                    else:
                        res = net(torch.Tensor(patch_resized).float().unsqueeze_(0))
                    if res.cpu().data.numpy()[0][0]>res.cpu().data.numpy()[0][1]:# test the patch and if it is a false positive add it to patches
                        patches.append(patch)

        if len(patches)> max_patches:
            patches = np.asarray(patches)
            print("loaded pascal data: %s" % (patches.shape,))
            return patches

    patches = np.asarray(patches)
    print("loaded pascal data: %s" % (patches.shape,))
    return patches

EPOCHS = 10
BATCH_SIZE = 32
LR = 0.0005
TEST_SPLIT = .2
PASCAL_TO_AFLW_RATIO = 5
use_cuda = torch.cuda.is_available()
personDir = "D:/TAU/EX2/EX2_data/VOC2007/ImageSets/Main/person_train.txt"
pascalDir = "D:/TAU/EX2/EX2_data/VOC2007/JPEGImages/"
aflwDir = "D:/TAU/EX2/EX2_data/aflw/aflw_24.t7"

# initialize random seed
torch.initial_seed()

aflwDataTorch  = torchfile.load(aflwDir, force_8bytes_long = True)
aflwData = np.zeros((len(aflwDataTorch),3 ,24, 24),dtype=float)
for i in range(len(aflwDataTorch)):
    aflwData[i] = aflwDataTorch[i+1]
print('loaded aflwData data: (%s, %s, %s, %s)' %(len(aflwData),aflwData[1].shape[0],aflwData[1].shape[1],aflwData[1].shape[2]) )
print(type(aflwData))

# parse data for train/test
net12 = torch.load('12net.pth')
if use_cuda:
    net12 = net12.cuda()
# negative mine false positive pathces from pascal images - takes some time
X_pascal_neg_mine = pascal_negative_mining(net12,personDir,pascalDir,net_input_dim=12,patch_dim=24, max_patches=len(aflwData)*PASCAL_TO_AFLW_RATIO,use_cuda = use_cuda)
y_pascal_neg_mine = np.ones((X_pascal_neg_mine.shape[0],1),dtype=int)
X_aflw = aflwData
y_aflw = np.zeros((X_aflw.shape[0],1),dtype=int)

X = np.concatenate((X_aflw,X_pascal_neg_mine),0)
y = np.concatenate((y_aflw,y_pascal_neg_mine),0)

length_data = X.shape[0]
shuffled_indices = torch.randperm(length_data)
split = int(np.floor(TEST_SPLIT * length_data))
indices_train = shuffled_indices[split:]
indices_val = shuffled_indices[:split]
X_train, X_test = X[indices_train], X[indices_val]
y_train, y_test = y[indices_train], y[indices_val]
X_test = torch.Tensor(X_test).float()
y_test = torch.Tensor(y_test).long()
dataSetTrain = aflwDataSet(X_train,y_train)
train_loader = DataLoader(dataset=dataSetTrain, batch_size=BATCH_SIZE, shuffle=True)

# define the model
net24 = torch.nn.Sequential(
          torch.nn.Conv2d(3, 64, 5,stride=1),
          torch.nn.MaxPool2d(3, stride=2),
          torch.nn.ReLU(),
          View(),
          torch.nn.Linear(5184, 128),
          torch.nn.ReLU(),
          torch.nn.Linear(128, 2),
        )
if use_cuda:
    net24.cuda()
print(summary(net24,(3,24,24)))

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net24.parameters(),lr=LR)

# TRAIN
loss_train_arr = []
loss_val_arr = []
print("{0:15} {1:20} {2:20}".format('EPOCHS','Train loss','Test loss'))
fig = plt.figure()
for ep_n, epoch in enumerate(range(EPOCHS)):
    running_loss = 0
    # run on train set
    for data in train_loader:
        inputs, targets = data
        if len(targets.size())>1:
            targets = torch.squeeze(targets,dim=1)

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = net24(inputs)

        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # run on test set
    with torch.set_grad_enabled(False):
        if len(y_test.size())>1:
            y_test = torch.squeeze(y_test, dim=1)
        if use_cuda:
            X_test = X_test.cuda()
            y_test = y_test.cuda()
        outputs = net24(X_test)
        loss_val = loss_fn(outputs, y_test)
        loss_val_arr.append(loss_val.cpu().data.numpy())

    loss_train_arr.append(running_loss/len(train_loader))
    print("{0:3d} {1:20f} {2:20f} ".format(ep_n, running_loss/len(train_loader),loss_val.cpu().data.numpy() ))

    plt.cla()
    plt.plot(loss_val_arr, linewidth=3)
    plt.plot(loss_train_arr, linewidth=3)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['test', 'train'], loc='upper right')
    plt.pause(0.01)

_, predicted = torch.max(outputs.data.cpu(), 1)
print('Network accuracy %d %%' % (100 * torch.sum(y_test.data.cpu() == predicted) / len(outputs)))

fig.savefig("q3.png")
# save model
torch.save(net24, "24net.pth")




