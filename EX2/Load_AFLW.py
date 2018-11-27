import torchfile
from matplotlib import pyplot as plt
import numpy as np

# load data
dataDir = "C:/Users/user007/Desktop/shani/CS/EX2_data/EX2_data/aflw/aflw_12.t7"
load_file = torchfile.load(dataDir, force_8bytes_long = True)

# show image
a = np.moveaxis(load_file[2], 0, -1)
print(a.shape)
plt.imshow(a)
plt.show()