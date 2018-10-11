#! /usr/bin/env python

import os,sys
import numpy as np
import glob
import random
from PIL import Image
import matplotlib.pyplot as plt


outfile = "./image/photos.npz"
max_photo = 10

photos = np.load(outfile)
xx = photos["x"]
yy = photos["y"]
idx = 0

print len(xx)
print len(yy)

plt.figure(figsize=(10,10))
for i in range(len(xx)):
    plt.subplot(4, max_photo, i+1)
    plt.title(yy[i+idx])
    plt.imshow(xx[i+idx])

plt.show()


