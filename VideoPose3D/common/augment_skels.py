import numpy as np
from common.debug import *

def augment_skels(skels, p):
    nSeq, nSkels , nJoints, nFeatures = skels.shape
    jointPermute = np.random.permutation(nJoints)
    skelPermute = np.random.permutation(nSkels)[0:round(p*nSkels)]
    skels[:, skelPermute, :, :] = skels[:, skelPermute, :, :][:, :, jointPermute, :]

    return skels, skelPermute