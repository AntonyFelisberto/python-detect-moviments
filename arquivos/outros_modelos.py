import numpy as np
import cv2
import sys
from random import randint

TEXT_COLOR = (randint(0,255),randint(0,255),randint(0,255))
BORDER_COLOR = (randint(0,255),randint(0,255),randint(0,255))
FONT = cv2.FONT_HERSHEY_COMPLEX
VIDEO_SOURCE = "arquivos\\cars\\cars.mp4"

BGS_TYPES = ["GMC","MOG2","MOG","KNN","CNT"]

def get_kernel(KERNEL_TYPE):
    if KERNEL_TYPE == "dilation":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    if KERNEL_TYPE == "opening":
        kernel = np.ones((3,3),np.uint8)
    if KERNEL_TYPE == "closing":
        kernel = np.ones((3,3),np.uint8)

    return kernel

def get_filter(img,filter):
    if filter == "clossing":
        return cv2.morphologyEx(img,cv2.MORPH_CLOSE,get_kernel("clossing"),iterations=2)
    if filter == "opening":
        return cv2.morphologyEx(img,cv2.MORPH_OPEN,get_kernel("opening"),iterations=2)
    if filter == "dilation":
        return cv2.morphologyEx(img,get_kernel("dilation"),iterations=2)
    if filter == "combine":
        closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,get_kernel("clossing"),iterations=2)
        opening = cv2.morphologyEx(closing,cv2.MORPH_OPEN,get_kernel("opening"),iterations=2)
        dilation = cv2.morphologyEx(opening,get_kernel("dilation"),iterations=2)
        return dilation
    
def get_bg_subtractor(BGS_TYPE):
    if BGS_TYPE == "GMG":
        cv2.bgsegm.createBackgroundSubtractorGMG(initiationFrames=120,decisionThreshold=0.8)
    if BGS_TYPE == "MOG":
        cv2.bgsegm.createBackgroundSubtractorMOG(history=200,nmixtures=5,backgroundRatio=0.7,noiseSigma=0)
    if BGS_TYPE == "MOG2":
        cv2.createBackgroundSubtractorMOG2(history=500,detectShadows=True,varThreshold=100)
    if BGS_TYPE == "KNN":
        cv2.createBackgroundSubtractorKNN(history=500,dist2Threshold=400,detectShadows=True)
    if BGS_TYPE == "CNT":
        cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15,useHistory=True,maxPixelStability=15*60,isParallel=True)
    
    