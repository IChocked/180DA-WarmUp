#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

'''
REFERENCES:
Masking/filtering image citation: https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
- No improvments made

Bounding box citation: https://stackoverflow.com/questions/23398926/drawing-bounding-box-around-given-size-area-contour
- changed code to only bound the largest detected object instead of all objects

Cropping an image in openCV: https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python
- nothing changed here

How to make a live graph with Matpotlib: https://www.geeksforgeeks.org/how-to-update-a-plot-in-matplotlib/
- changed to accomodate the histogram

K clusters: https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
- didn't change anything
'''

def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

cap = cv.VideoCapture(0)

plt.ion()
fig = plt.figure()

while(1):
    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # define range of red color in HSV
    lower_red = np.array([170, 16,16])
    upper_red = np.array([360,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_red, upper_red)

    # Bitwise-AND mask and original image
    #res = cv.bitwise_and(frame,frame, mask= mask)

    contours, _ = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0
    cont = -1
    for i in range(0, len(contours)):
        c = contours[i]
        rect = cv.boundingRect(c)
        if rect[2] < 100 or rect[3] < 100: continue
        if cv.contourArea(c) > max_area:
            max_area = cv.contourArea(c)
            cont = c

    x, y, w, h = -1, -1, -1, -1
    if type(cont) != int and type(cont) != float:
        rect = cv.boundingRect(cont)
        x,y,w,h = rect
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv.putText(frame,'Water bottle!!!',(x+w+10,y+h),0,0.3,(0,255,0))

    cv.imshow('frame',frame)
    cv.imshow('mask',mask)

    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

    if x == -1 or y == -1 or w == -1 or h == -1:
        continue
    
    img = frame[y:y+h, x:x+w]
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
    clt = KMeans(n_clusters=3) #cluster number
    clt.fit(img)

    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)

    plt.axis("off")
    plt.imshow(bar)
    fig.canvas.draw()
    fig.canvas.flush_events()


cv.destroyAllWindows()
