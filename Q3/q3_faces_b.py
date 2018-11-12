
import os
import torch
import pandas as pd
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision
import pickle
from torchsummary import summary

# load data
dataDir = "drive/My Drive/Colab Notebooks/TAU_2018_DL/HW1/Q3"

FTRAIN = dataDir + "/training.csv"
FTEST = dataDir + "/test.csv"


def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    #print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

# load train data
X, y = load()

# change numpy array to torch
X_input = torch.from_numpy(X).float()
y_output = torch.from_numpy(y).float()

saveName_trainedModel = dataDir + "/Q3b_trainedModel.pth"
isLoadNet = False
if isLoadNet:
  net = torch.load('Q3b_trainedModel.pth')

use_cuda = torch.cuda.is_available()
if use_cuda:
    X_input = X_input.cuda()
    y_output = y_output.cuda()

class View(torch.nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

# split dataset for train and validation
validation_split = .2
length_data = X_input.shape[0] 

# set seed for random indices of data (for debugging)
torch.manual_seed(1)

shuffled_indices = torch.randperm(length_data)
  
split = int(np.floor(validation_split * length_data))
indices_train = shuffled_indices[split:]
indices_val = shuffled_indices[:split]
X_train, X_val = X_input[indices_train], X_input[indices_val]
y_train, y_val = y_output[indices_train], y_output[indices_val]

# initialize random seed
torch.initial_seed()

print(X_train.size(), y_train.size())
print(X_val.size(), y_val.size())

# reshape val & train
X_val = X_val.reshape([-1,1,96,96])
X_train = X_train.reshape([-1,1,96,96])
  
trainset = torch.utils.data.TensorDataset(X_train,y_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

# check dims    
for i,data in enumerate(trainloader,0):
    # get the inputs
    inputs, labels = data
    print(inputs.shape, labels.shape)
    
print("number of batches is " + str(len(trainloader)))



# define the model
net = torch.nn.Sequential(
          torch.nn.Conv2d(1, 32, 3),
          torch.nn.MaxPool2d(2),
          torch.nn.ReLU(),
          torch.nn.Conv2d(32, 64, 2),
          torch.nn.MaxPool2d(2),
          torch.nn.ReLU(),
          torch.nn.Conv2d(64, 128, 2),
          torch.nn.MaxPool2d(2),
          torch.nn.ReLU(),
          View(),
          torch.nn.Linear(15488, 500),
          torch.nn.ReLU(),
          torch.nn.Linear(500, 500),
          torch.nn.ReLU(),
          torch.nn.Linear(500, 30),
        )

if use_cuda:
    net = net.cuda()

loss_fn = torch.nn.MSELoss()
optimizer = optim.Adam(net.parameters(),lr=0.01)

print(summary(net, (1,96,96)))

EPOCHS = 1000
nSamplesTrain = X_train.shape[0] 
loss_train_arr = np.zeros(EPOCHS)
loss_val_arr = np.zeros(EPOCHS)

print("{0:15} {1:20} {2:20}".format('EPOCHS','Train loss','Validation loss'))

for t in range(EPOCHS):
    running_loss = 0.0
    
    with torch.set_grad_enabled(False):
        net.eval()
        y_val_pred = net(X_val)
        loss_val = loss_fn(y_val_pred,y_val)
        loss_val_arr[t] = loss_val.cpu().data.numpy()
        
        
    # train network
    net.train()
    
    nBatches = 0;
    for i,data in enumerate(trainloader,0):
        # get the inputs
        inputs, labels = data
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    loss_train_arr[t] = running_loss / len(trainloader)

    print("{0:3d} {1:20f} {2:20f}".format(t,running_loss / len(trainloader),loss_val.item()))

# save model
torch.save(net, saveName_trainedModel)

# save model
torch.save(net, saveName_trainedModel)

# plot loss for train and validation
startIdx = 10
saveName = dataDir + "/training_Q3b.png"
xi = [i for i in range(startIdx, EPOCHS)]

fig = plt.figure()
plt.plot(xi, loss_val_arr[startIdx::], linewidth=3)
plt.plot(xi, loss_train_arr[startIdx::], linewidth=3)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['test', 'train'], loc='upper right')
plt.show()
fig.savefig(saveName)

print("loss function value in last epoch is: test:{:f} train:{:f}".format(loss_train_arr[-1].item(), loss_val_arr[-1].item()))

# print convergence in comparison to previous network
trainSaveName_previous = dataDir + "/Q3a_TrainResults.pickle"
testSaveName_previous = dataDir + "/Q3a_TestResults.pickle"
with open(trainSaveName_previous, 'rb') as f:
    loss_train_previous = pickle.load(f)
with open(testSaveName_previous, 'rb') as f:
    loss_test_previous = pickle.load(f)
    
# plot loss for train and validation
startIdx = 10
saveName = dataDir + "/comparison_Q3b.png"
xi = [i for i in range(startIdx, EPOCHS)]

fig = plt.figure()
plt.plot(xi, loss_val_arr[startIdx::], linewidth=3)
plt.plot(xi, loss_train_arr[startIdx::], linewidth=3)
plt.plot(xi, loss_test_previous[startIdx::], linewidth=3)
plt.plot(xi, loss_train_previous[startIdx::], linewidth=3)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['test CNN', 'train CNN', 'test FC', 'train FC', ], loc='upper right')
plt.show()
fig.savefig(saveName)

net.eval()
# plot location of organs
def plot_sample(x, y, imgTitle, axis):
    #img = x.reshape(96, 96)
    img = x
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=20, color = "r")
    axis.set_title(imgTitle)
    axis.titlesize : large   # fontsize of the axes title
	
sampleID = 2;
# change to numpy array
sampleX = X_val[sampleID,:].cpu().data.numpy()
sampley = y_val[sampleID,:].cpu().data.numpy()
# remove first dimension
sampleX = np.squeeze(sampleX)

fig = plt.figure(figsize=(12, 24))

# plot ground truth
ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
plot_sample(sampleX, sampley, 'Ground Truth', ax)


# run in network
x_val_current = X_val[sampleID,:]
# add batch dimension
x_val_current = x_val_current.unsqueeze_(0)
# run in network
y_pred = net(x_val_current)
# change to numpy array for plotting
y_pred = y_pred.cpu().data.numpy()
# remove batch dimension
y_pred = np.squeeze(y_pred)

# plot network result
ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
plot_sample(sampleX, y_pred, 'Net\'s Output', ax)