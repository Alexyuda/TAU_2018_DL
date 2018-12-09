# isOnColab = True
#
# if isOnColab:
#   from google.colab import drive
#   drive.mount('/content/drive/')
#   !pip3 install torch torchvision torchfile torchsummary pillow
#   !pip3 uninstall Pillow -y
#   !pip3 install Pillow==4.1.1

# check that notebook can access gDrive
# ls drive/My\ Drive/Colab\ Notebooks/TAU_2018_DL/HW2/EX2_data

# get dataset
# data_main = "C:/Users/Shani/Downloads/DL - CS\EX2/EX2_data"

# downloadVOCData = False
# if downloadVOCData:
#   !wget http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar -P drive/My\ Drive/Colab\ Notebooks/TAU_2018_DL/HW2/EX2_data
#   !mkdir drive/My\ Drive/Colab\ Notebooks/TAU_2018_DL/HW2/EX2_data/VOC2007
#   !tar -xf drive/My\ Drive/Colab\ Notebooks/TAU_2018_DL/HW2/EX2_data/VOCtrainval_06-Nov-2007.tar -C drive/My\ Drive/Colab\ Notebooks/TAU_2018_DL/HW2/EX2_data/VOC2007
#   !rm -rf drive/My\ Drive/Colab\ Notebooks/TAU_2018_DL/HW2/EX2_data/VOC2007/VOCtrainval_06-Nov-2007.tar
#
# downloadFDDB = False
# if downloadFDDB:
#   !wget http://tamaraberg.com/faceDataset/originalPics.tar.gz -P drive/My\ Drive/Colab\ Notebooks/TAU_2018_DL/HW2/EX2_data
#   !mkdir drive/My\ Drive/Colab\ Notebooks/TAU_2018_DL/HW2/EX2_data/FDDB
#   !tar -xf drive/My\ Drive/Colab\ Notebooks/TAU_2018_DL/HW2/EX2_data/originalPics.tar.gz -C drive/My\ Drive/Colab\ Notebooks/TAU_2018_DL/HW2/EX2_data/FDDB
#   !rm -rf drive/My\ Drive/Colab\ Notebooks/TAU_2018_DL/HW2/EX2_data/originalPics.tar.gz

import torch
import torchfile
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction import image
import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import torch.optim as optim
from PIL import Image
from skimage import transform
import cv2
import os


class View(torch.nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


# Non Maximum supression
def NMS(boxes):
    if len(boxes) == 0:
        return []

    # sort boxes according to score
    # x_min, y_min, x_max, y_max, score
    boxes.sort(key=lambda val: val[4])
    boxes.reverse()

    x_min = np.array([boxes[i][0] for i in range(len(boxes))], np.float32)
    y_min = np.array([boxes[i][1] for i in range(len(boxes))], np.float32)
    x_max = np.array([boxes[i][2] for i in range(len(boxes))], np.float32)
    y_max = np.array([boxes[i][3] for i in range(len(boxes))], np.float32)

    area = (x_max - x_min) * (y_max - y_min)
    idxs = np.array(range(len(boxes)))
    savedB = []
    while len(idxs) > 0:
        # compare to box with highest score
        i = idxs[0]
        savedB.append(i)

        # intersection with all boxes
        xx1 = np.maximum(x_min[i], x_min[idxs[1:]])
        yy1 = np.maximum(y_min[i], y_min[idxs[1:]])
        xx2 = np.minimum(x_max[i], x_max[idxs[1:]])
        yy2 = np.minimum(y_max[i], y_max[idxs[1:]])

        w = np.maximum(xx2 - xx1, 0)
        h = np.maximum(yy2 - yy1, 0)

        overlap = (w * h) / (area[idxs[1:]] + area[i] - w * h)

        # remove boxes that intersect with current higher score box
        idxs = np.delete(idxs, np.concatenate(([0], np.where(((overlap >= 0.3) & (overlap <= 1)))[0] + 1)))

    return [boxes[i] for i in savedB]


# save net output to boxes
# stride is the ratio between net result and image (x2 downsample for example)
def createBoxes(netResult, stride, winHalfSize=6):
    width = int(stride * netResult.shape[1])
    height = int(stride * netResult.shape[0])

    # print(netResult.shape)

    isFace = netResult[:, :, 0] > netResult[:, :, 1]

    boxes = []

    for i in range(0, netResult.shape[0]):
        for j in range(0, netResult.shape[1]):
            if (isFace[i, j]):
                row = int(stride * i - (stride / 2.0))
                col = int(stride * j + (stride / 2.0))
                x_min = np.maximum(0, col - winHalfSize)
                y_min = np.maximum(0, row - winHalfSize)
                x_max = np.minimum(width, col + winHalfSize)
                y_max = np.minimum(height, row + winHalfSize)
                score = netResult[i, j, 0]

                boxes.append([x_min, y_min, x_max, y_max, score])

    return boxes


def drawBoxes(image, boxes):
    img = image.copy()
    img = 255 * img  # Now scale by 255
    img = img.astype(np.uint8)

    for i in range(len(boxes)):
        x_min = boxes[i][0]
        y_min = boxes[i][1]
        x_max = boxes[i][2]
        y_max = boxes[i][3]

        img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

    return img


def runNetMulScales(image, net, scales, winHalfSize=6, heatStride=2):
    boxes = []
    for idxScale in range(0, len(scales)):
        # resize image
        curScale = scales[idxScale]

        img_resized = transform.resize(image, (int(img.shape[0] * curScale), int(img.shape[1] * curScale)))

        # run in net
        img_resized_forNet = np.moveaxis(img_resized, -1, 0)
        if use_cuda:
            img_resized_forNet = torch.Tensor(img_resized_forNet).float().cuda()
        y_val = net(img_resized_forNet.unsqueeze_(0))
        y_val = torch.squeeze(y_val)

        heatMap = y_val.cpu().data.numpy()
        heatMap = np.moveaxis(heatMap, 0, -1)

        # create boxes from result and run NMS
        stride = (1.0 / curScale) * heatStride

        winHalfSizeCur = int((1.0 / curScale) * winHalfSize)
        curBoxes = createBoxes(heatMap, stride, winHalfSizeCur)
        curBoxes = NMS(curBoxes)
        boxes = boxes + curBoxes

    return boxes


use_cuda = torch.cuda.is_available()

# set FCN
net = torch.nn.Sequential(
    torch.nn.Conv2d(3, 64, 5, stride=1, padding=12),
    torch.nn.MaxPool2d(3, stride=2),
    torch.nn.ReLU(),
    torch.nn.Conv2d(64, 128, 9),
    torch.nn.ReLU(),
    torch.nn.Conv2d(128, 2, 1),
    )

if use_cuda:
    net.cuda()
    print("using cuda!")

print(summary(net, (3, 24, 24)))

# load pre-trained weights
# modelPath = data_main + "/12net.pth"
modelPath = "C:/Users/Shani/PycharmProjects/TAU_2018_DL/EX2/Q3/24net.pth"
pre_trained_model = torch.load(modelPath)

net[0].weight = pre_trained_model[0].weight
net[0].bias = pre_trained_model[0].bias
fc4 = pre_trained_model[4].state_dict()
net[3].load_state_dict({"weight": fc4["weight"].view(128, 64, 9, 9), "bias": fc4["bias"]})
fc6 = pre_trained_model[6].state_dict()
net[5].load_state_dict({"weight": fc6["weight"].view(2, 128, 1, 1), "bias": fc6["bias"]})

data_main = "C:/Users/Shani/Downloads/DL - CS/EX2/EX2_data"
data_aflw = data_main + "/aflw/aflw_12.t7"
data_pascal_imgs = data_main + "/PASCAL/VOC2007/JPEGImages/"
data_pascal_personIdx = data_main + "/PASCAL/VOC2007/ImageSets/Main/person_train.txt"
data_fddb = data_main + "/fddb"

# load pascal data
personDir = data_pascal_personIdx  # "D:/TAU/EX2/EX2_data/VOC2007/ImageSets/Main/person_train.txt"
pascalDir = data_pascal_imgs  # "D:/TAU/EX2/EX2_data/VOC2007/JPEGImages/"

# load all images with a person
myarray = np.loadtxt(personDir)
myarray_new = np.zeros(myarray.shape[0])
j = 0
for i, line in enumerate(myarray):
    if (line[1] == 1):
        myarray_new[j] = line[0]
        j += 1
    pos_file_array = myarray_new[:j]

nSamples = len(pos_file_array)

# run network on sample image
sampleID = 103
heatStride = 2
# create figure
rszScale = 0.25

# load image and resize
filename = str(int(pos_file_array[sampleID])).zfill(6) + '.jpg'
img = mpimg.imread(pascalDir + filename) / 255
img_resized = transform.resize(img, (int(img.shape[0] * rszScale), int(img.shape[1] * rszScale)))

img_resized_forNet = np.moveaxis(img_resized, -1, 0)
if use_cuda:
    img_resized_forNet = torch.Tensor(img_resized_forNet).float().cuda()

y_val = net(img_resized_forNet.unsqueeze_(0))
y_val = torch.squeeze(y_val)
heatMap = y_val.cpu().data.numpy()
heatMap = np.moveaxis(heatMap, 0, -1)

# create & draw boxes
winHalfSize = 6
stride = 2
boxes = createBoxes(heatMap, stride, winHalfSize)
img_colored2 = drawBoxes(img_resized, boxes)

fig = plt.figure(figsize=(12, 24))

# show image
ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
ax.imshow(img_colored2)
ax.set_title("boxes BEFORE NMS")

# preform Non-Maximum-Sup. and draw boxes
boxes_NMS = NMS(boxes)
img_colored_NMS = drawBoxes(img_resized, boxes_NMS)
ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
ax.imshow(img_colored_NMS)
ax.set_title("boxes AFTER NMS")

scales = [0.2, 1]

fig = plt.figure(figsize=(10, 10))
boxes = runNetMulScales(img, net, scales)
img_colored = drawBoxes(img, boxes)
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
ax.imshow(img_colored)
plt.show()

# run test on FDDB

filename_results = data_main + "/fold-01-out.txt"
fileFDDB_res = open(filename_results, "w")
scales = [0.05, 0.1, 0.2, 0.4, 0.5]

fileNames_fddb = data_fddb + "/FDDB-folds/FDDB-fold-01.txt"
with open(fileNames_fddb) as fp:
    lines = fp.readlines()

for i, line in enumerate(lines):
    line = line[:-1]  # remove \n
    filename = data_fddb + "/images/" + line + ".jpg"
    curImg = mpimg.imread(filename) / 255

    # if gray scale image copy gray channel to create "RGB" image
    if curImg.ndim == 2:
        curImgrgb = np.concatenate((curImg[..., None], curImg[..., None], curImg[..., None]), axis=2)
        curImg = curImgrgb

    boxes = runNetMulScales(curImg, net, scales)

    # write to file
    fileFDDB_res.write("%s\n" % line)  # image name
    nBoxes = len(boxes)
    fileFDDB_res.write("%d\n" % nBoxes)  # number of boxes

    # change to ellipse
    for i in range(0, nBoxes):
        x_min = boxes[i][0]
        y_min = boxes[i][1]
        x_max = boxes[i][2]
        y_max = boxes[i][3]

        # Here, each face is denoted by:
        # <major_axis_radius minor_axis_radius angle center_x center_y 1>.
        minor_axis_radius = (x_max - x_min) / 2.0
        major_axis_radius = (y_max - y_min) / 2.0
        angle = 0
        center_x = (x_max + x_min) / 2.0
        center_y = (y_max + y_min) / 2.0

        elipse_text = '%.6f %.6f %.6f %.6f %.6f 1' % (
        major_axis_radius, minor_axis_radius, angle, center_x, center_y)  # 3.1+ only
        fileFDDB_res.write("%s\n" % elipse_text)  # current face

    # img_colored = drawBoxes(curImg, boxes)
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
    # ax.imshow(img_colored)

fileFDDB_res.close()

# !drive/My\ Drive/Colab\ Notebooks/TAU_2018_DL/HW2/EX2_data/fddb/evaluation/runEvaluate.pl drive/My\ Drive/Colab\ Notebooks/TAU_2018_DL/HW2/EX2_data/