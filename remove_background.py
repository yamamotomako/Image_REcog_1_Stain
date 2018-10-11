#! /usr/bin/env python

import os,sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


x = []
y = []
p_h_size = 150
p_w_size = 250

def convert(folder):
    pnglist = os.listdir("./image_new/"+folder)

    #canvas = Image.new("RGB", (150,150), (255,255,255))

    for png in pnglist:
        img = Image.open("./image_new/"+folder+"/"+png)

        width, height = img.size
        img2 = Image.new("RGB",(width,height))

        print height, width
        print png

        #img_pixels = []
        #for y in range(height):
        #    for x in range(width):
        #        img_pixels.append(img.getpixel((x,y)))

        #img_pixels = np.array(img_pixels)
        #print len(img_pixels)

        #print img_pixels[:,3]
        #print img_pixels[:,[0,1,2]]
        #print img_pixels[100][200]

        for i in range(height):
            for j in range(width):
                buf = img.getpixel((j,i))
                #print buf
                r = int(buf[0])
                g = int(buf[1])
                b = int(buf[2])
                #print r,g,b

                if r > 140 and g > 78 and b > 100:
                    img2.putpixel((j,i),(255,255,255,255))
                else:
                    img2.putpixel((j,i),(r,g,b,255))

        #img2.show()
        img2.save("./image_conv/"+folder+"/"+png, quality=30)

        index = pnglist.index(png)
        print index

        #canvas.paste(img.resize((200,200)), (0,50*index))
        #canvas.paste(img2.resize((200,200)), (50,50*index))

    #canvas.save("./all.png", "PNG", quality=100)
    #canvas.show()


def paste(folder, label):
    pnglist = os.listdir("./image_new/"+folder)
    plt.figure(figsize=(20,20))

    for png in pnglist:
        img = Image.open("./image_new/"+folder+"/"+png)
        img = img.convert("RGB")
        img = img.resize((p_w_size, p_h_size))
        img = np.asarray(img)

        img2 = Image.open("./image_conv/"+folder+"/"+png)
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
    plt.savefig("./"+folder+".png")

    



convert("AML")
convert("MDS")

paste("AML", 0)
paste("MDS", 1)

np.savez("./all.npz", x=x, y=y)
