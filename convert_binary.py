#! /usr/bin/python

import os,sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


x = []
y = []
p_h_size = 150
p_w_size = 250

def convert(folder):
    pnglist = os.listdir("./image_conv/"+folder)

    #canvas = Image.new("RGB", (150,150), (255,255,255))

    for png in pnglist:
        img = Image.open("./image_conv/"+folder+"/"+png)

        width, height = img.size
        img2 = Image.new("RGB",(width,height))

        #print height, width
        print png

        for i in range(height):
            for j in range(width):
                buf = img.getpixel((j,i))
                r = int(buf[0])
                g = int(buf[1])
                b = int(buf[2])

                if r != 255 and g != 255 and b != 255:
                    img2.putpixel((j,i),(0,0,0,255))
                else:
                    img2.putpixel((j,i),(255,255,255,255))

        #img2.show()
        img2.save("./image_binary/"+folder+"/"+png, quality=30)

        index = pnglist.index(png)
        print index



def paste(folder, label):
    pnglist = os.listdir("./image_conv/"+folder)
    plt.figure(figsize=(20,20))

    for png in pnglist:
        img = Image.open("./image_conv/"+folder+"/"+png)
        img = img.convert("RGB")
        img = img.resize((p_w_size, p_h_size))
        img = np.asarray(img)

        img2 = Image.open("./image_binary/"+folder+"/"+png)
        img2 = img2.convert("RGB")
        img2 = img2.resize((p_w_size, p_h_size))
        img2 = np.asarray(img2)

        index = pnglist.index(png)

        x.append(img)
        #x.append(img2)
        y.append(label)

        plt.subplot(10, 10, index*2+1)
        plt.title(png)
        plt.imshow(img)
        plt.subplot(10, 10, index*2+2)
        plt.title(png)
        plt.imshow(img2)

    #plt.show()
    plt.savefig("./"+folder+"_binary.png")

    



convert("AML")
convert("MDS")

paste("AML", 0)
paste("MDS", 1)

np.savez("./all_binary.npz", x=x, y=y)
