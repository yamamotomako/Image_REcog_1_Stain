#! /usr/bin/env python

import os,sys
import numpy as np
import glob
import random
from PIL import Image
import matplotlib.pyplot as plt


outfile = "./image/photos.npz"
max_photo = 3
photo_size = 100
x = []
y = []

def main():
    glob_files("./image_new/AML",0)
    #glob_files("./image/AML/8-21", 1)
    glob_files("./image_new/B-cell", 1)
    #glob_files("./image/B-cell/zenku", 3)
    #glob_files("./image/MDS/5q", 4)
    glob_files("./image_new/T-NK", 2)
    glob_files("./image_new/tenni", 3)

    np.savez(outfile, x=x, y=y)
    print "saved"

    photos = np.load(outfile)
    xx = photos["x"]
    yy = photos["y"]
    idx = 0

    print len(xx)
    print len(yy)

    plt.figure(figsize=(15,15))
    for i in range(len(xx)):
        plt.subplot(4, max_photo, i+1)
        plt.title(yy[i+idx])
        plt.imshow(xx[i+idx])

    plt.show()


def glob_files(path, label):
    files = glob.glob(path + "/*.png")
    random.shuffle(files)
    num = 0
    
    for f in files:
        if num >= max_photo: break
        num += 1
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((photo_size, photo_size))
        img = np.asarray(img)
        x.append(img)
        y.append(label)


if __name__ == "__main__":
    main()


