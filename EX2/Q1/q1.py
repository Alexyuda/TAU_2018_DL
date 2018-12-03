import torch
import torchfile
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction import image
import matplotlib.image as mpimg
from torch.utils.data import Dataset,DataLoader
from torchsummary import summary
import torch.optim as optim


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


def create_PASCAL_imgs(personDir,pascalDir,dim, max_patches):

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

    # Create random patches:
    # Every patch line in patches describes a patch with dimensions = (12,12,3)
    #(total lines = max_patches * len(neg_file_array))
    patches = np.zeros((max_patches * len(neg_file_array), dim, dim, 3),dtype=float)

    # create patches
    i = 0
    for line in neg_file_array:
        filename = str(int(line)).zfill(6) + '.jpg'
        img = mpimg.imread(pascalDir + filename)/255
        img_patches = image.extract_patches_2d(img, (dim, dim), max_patches=max_patches)
        patches[i:i + max_patches] = img_patches
        # plt.imshow(img_patches[14])
        # plt.show()
        # plt.imshow(patches[i + 14])
        # plt.imshow(patches[i + 5])
        i += max_patches

    # plt.imshow(patches[14])
    # plt.show()

    # Change dimensions to (total patches, 3, dim, dim)
    patches = np.moveaxis(patches, -1, 1)
    print("loaded pascal data: %s" % (patches.shape,))
    return patches

EPOCHS = 200
BATCH_SIZE = 256
LR = 0.0005
TEST_SPLIT = .2
PASCAL_TO_AFLW_RATIO = 3

use_cuda = torch.cuda.is_available()

# load aflw Data
aflwDir = "D:/TAU/EX2/EX2_data/aflw/aflw_12.t7"
aflwDataTorch  = torchfile.load(aflwDir, force_8bytes_long = True)
aflwData = np.zeros((len(aflwDataTorch),3 ,12, 12),dtype=float)
for i in range(len(aflwDataTorch)):
    aflwData[i] = aflwDataTorch[i+1]
print('loaded aflwData data: (%s, %s, %s, %s)' %(len(aflwData),aflwData[1].shape[0],aflwData[1].shape[1],aflwData[1].shape[2]) )
print(type(aflwData))
# plt.imshow( np.moveaxis(aflwData[0], 0, -1))
# plt.show()

# load pascal data
personDir = "D:/TAU/EX2/EX2_data/VOC2007/ImageSets/Main/person_train.txt"
pascalDir = "D:/TAU/EX2/EX2_data/VOC2007/JPEGImages/"
pascalData = create_PASCAL_imgs(personDir,pascalDir,12, 60)
print(type(pascalData))

# build data loader for aflw
length_data = aflwData.shape[0]
shuffled_indices = torch.randperm(length_data)
split = int(np.floor(TEST_SPLIT * length_data))
indices_train = shuffled_indices[split:]
indices_val = shuffled_indices[:split]
X_train_aflw, X_test = aflwData[indices_train], aflwData[indices_val]
X_train_pascal = pascalData[0:min(len(aflwData)*PASCAL_TO_AFLW_RATIO,pascalData.shape[0]-1)]

X_test = torch.Tensor(X_test).float()
y_val_target = torch.Tensor(np.zeros(X_test.shape[0])).long()

y_train_aflw = np.zeros((X_train_aflw.shape[0],1),dtype=int)
y_train_pascal = np.ones((X_train_pascal.shape[0],1),dtype=int)

X_train = np.concatenate((X_train_aflw,X_train_pascal),0)
y_train = np.concatenate((y_train_aflw,y_train_pascal),0)

dataSetTrain = aflwDataSet(X_train,y_train)
train_loader = DataLoader(dataset=dataSetTrain, batch_size=BATCH_SIZE, shuffle=True)

# define the model
net = torch.nn.Sequential(
          torch.nn.Conv2d(3, 16, 3,stride=1),
          torch.nn.MaxPool2d(3, stride=2),
          torch.nn.ReLU(),
          View(),
          torch.nn.Linear(256, 16),
          torch.nn.ReLU(),
          torch.nn.Linear(16, 2),
        )

if use_cuda:
    net.cuda()

print(summary(net,(3,12,12)))

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=LR)

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
        targets = torch.squeeze(targets,dim=1)

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # run on test set
    with torch.set_grad_enabled(False):
        if use_cuda:
            X_test = X_test.cuda()
            y_val_target = y_val_target.cuda()
        y_val = net(X_test)
        loss_val = loss_fn(y_val, y_val_target)
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

_, predicted = torch.max(y_val.data.cpu(), 1)
print('Network accuracy %d %%' % (100 * torch.sum(y_val_target.data.cpu() == predicted) / len(y_val)))

fig.savefig("q1.png")
# save model
torch.save(net, "12net.pth")






