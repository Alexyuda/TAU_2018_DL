
import os
import torch
import pandas as pd
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.utils.data
import pickle

# load data
dataDir = []"drive/My Drive/Colab Notebooks/TAU_2018_DL/HW1/Q3"

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

use_cuda = torch.cuda.is_available()
if use_cuda:
    X_input = X_input.cuda()
    y_output = y_output.cuda()

# split dataset for train and validation

# split ratio
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

# define the model
net = torch.nn.Sequential(
          torch.nn.Linear(9216, 100),
          torch.nn.ReLU(),
          torch.nn.Linear(100, 30),
        )

if use_cuda:
    net = net.cuda()


loss_fn = torch.nn.MSELoss()
optimizer = optim.Adam(net.parameters(),lr=0.01)

# loss_fn = torch.nn.MSELoss(reduction='sum')

EPOCHS = 1000
nSamplesTrain = X_train.shape[0] 
loss_train_arr = np.zeros(EPOCHS)
loss_val_arr = np.zeros(EPOCHS)

print("{0:15} {1:20} {2:20}".format('EPOCHS','Train loss','Validation loss'))

# initialize progress bar
#out = display(progress(0, 100), display_id=True)
trainSaveName = dataDir + "/Q3a_TrainResults.pickle"
testSaveName = dataDir + "/Q3a_TestResults.pickle"
fileTrain  = open(trainSaveName, "w")
fileTest = open(testSaveName, "w")
net.train()

for t in range(EPOCHS):
  # evaluate network
  net.eval()
  y_val_pred = net(X_val)
  loss_val = loss_fn(y_val_pred,y_val)
  loss_val_arr[t] = loss_val.cpu().data.numpy()
  
  # train network
  net.train()
  
  # shuffle data
  shuffle = torch.randperm(nSamplesTrain)
  X_train = X_train[shuffle, :]
  y_train = y_train[shuffle, :]
  
  # forward pass
  y_pred = net(X_train)
  
  # compute forward pass and loss for train
  loss = loss_fn(y_pred, y_train) 
  loss_train_arr[t] = loss.cpu().data.numpy()
  
  # backward pass
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  
#   print(t,loss.item(),loss_val.item())
  print("{0:3d} {1:20f} {2:20f}".format(t,loss.item(),loss_val.item()))
  fileTrain.write("%f\n" % loss.cpu().data.numpy())
  fileTest.write("%f\n" % loss_val.cpu().data.numpy())
  #out.update(progress(t, EPOCHS))
  
fileTrain.close()
fileTest.close()

with open(trainSaveName, 'wb') as f:
    pickle.dump(loss_train_arr, f)
with open(testSaveName, 'wb') as f:
    pickle.dump(loss_val_arr, f)

# plot loss for train and validation
startIdx = 10
saveName = dataDir + "/training_Q3a.png"
xi = [i for i in range(startIdx, EPOCHS)]

fig = plt.figure()
plt.plot(xi, loss_val_arr[startIdx::], linewidth=3)
plt.plot(xi, loss_train_arr[startIdx::], linewidth=3)
# plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['test', 'train'], loc='upper right')
plt.show()
fig.savefig(saveName)

print("loss function value in last epoch is: test:{:f} train:{:f}".format(loss_val.item(), loss.item()))

# plot location of organs
def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=20, color = "r")

sampleID = 10;
# change to numpy array
sampleX = X_val[sampleID,:].cpu().data.numpy()
sampley = y_val[sampleID,:].cpu().data.numpy()

fig = plt.figure(figsize=(12, 24))

# plot ground truth
ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
plot_sample(sampleX, sampley, ax)


# run in network
y_pred = net(X_val[sampleID,:])
# change to numpy array for plotting
y_pred = y_pred.cpu().data.numpy()
ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
plot_sample(sampleX, y_pred, ax)
