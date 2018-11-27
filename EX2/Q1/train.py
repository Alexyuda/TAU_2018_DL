import numpy as np
from matplotlib import pyplot as plt
from Load_PASCAL import load_PASCAL_imgs


# Loading images
# Switch to dim = 12 or 24
PASCAL_imgs_12net = load_PASCAL_imgs(dim = 24)
print(type(PASCAL_imgs_12net))
print(PASCAL_imgs_12net.shape)

# Show image 1:
# Change axis:
image = np.moveaxis(PASCAL_imgs_12net[549], 0, -1)
print(image.shape)
plt.imshow(image)
plt.show()

