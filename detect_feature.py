#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os,sys
import numpy as np
import cv2
import matplotlib.pyplot as plt


img1 = cv2.imread('./image_new/2018-09-2515.52.33.png')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp1 = sift.detect(gray1)
img1_sift = cv2.drawKeypoints(gray1, kp1, None, flags=4)


