import numpy as np
from common.debug import *
import torch

def augment_skels(skels, p):
    nSeq, nSkels , nJoints, nFeatures = skels.shape
    randPerm = np.random.permutation(nSkels)
    skelPermute = randPerm[0:round(p*nSkels)]
    nonskelPermute = randPerm[round(p*nSkels):]
    if skelPermute.__len__()>0:
        randomTen = torch.randn(skels[:, skelPermute, :, :].shape)
        # if torch.cuda.is_available():
        #     randomTen = randomTen.cuda()
        #skels[:, skelPermute, :, :] = skels[:, skelPermute, :, :] * randomTen
        skels[:, skelPermute, :, :] = skels[:, skelPermute, :, :] * 0

    return skels, skelPermute, nonskelPermute

def interp_skels(skels,p):
    nSeq, nSkels , nJoints, nFeatures = skels.shape
    randPerm = np.random.permutation(nSkels)
    skelPermute = randPerm[0:round(p*nSkels)]
    nonskelPermute = randPerm[round(p * nSkels):]

    # interp here
    # calc dist to