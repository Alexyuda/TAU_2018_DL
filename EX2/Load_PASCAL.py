# This function creates two numpy array files with random patches
# One for 12net and the other for 24net

import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction import image
import matplotlib.image as mpimg

# directory of person_train.txt and JPEGImages folder
dir_person = "C:/Users/user007/Desktop/shani/CS/EX2_data/EX2_data/PASCAL/VOCdevkit/VOC2007/ImageSets/Main/person_train.txt"
dir_images = "C:/Users/user007/Desktop/shani/CS/EX2_data/EX2_data/PASCAL/VOCdevkit/VOC2007/JPEGImages/"

def create_PASCAL_imgs(dim, max_patches):

    # Create a list of images without a person
    #(in person_train.txt a line with label == -1 is a negative sample)
    myarray = np.loadtxt(dir_person)
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
    patches = np.zeros((max_patches * len(neg_file_array), dim, dim, 3),dtype=int)

    # create patches
    i = 0
    for line in neg_file_array:
        filename = str(int(line)).zfill(6) + '.jpg'
        img = mpimg.imread(dir_images + filename)
        img_patches = image.extract_patches_2d(img, (dim, dim), max_patches=max_patches)
        patches[i:i + max_patches] = img_patches
        # plt.imshow(img_patches[14])
        # plt.imshow(patches[i + 14])
        # plt.imshow(patches[i + 5])
        i += max_patches

    # plt.imshow(patches[14])
    # plt.show()

    # Change dimensions to (total patches, 3, dim, dim)
    patches = np.moveaxis(patches, -1, 1)
    print(patches.shape)

    filename = 'PASCAL_imgs_' + str(dim) + '.npy'
    np.save(filename, patches)


def load_PASCAL_imgs(dim):
    filename = 'PASCAL_imgs_' + str(dim) + '.npy'
    PASCAL_imgs = np.load(filename)
    return PASCAL_imgs


if __name__ == '__main__':
    create_PASCAL_imgs(dim = 12, max_patches = 15)
    create_PASCAL_imgs(dim = 24, max_patches = 15)