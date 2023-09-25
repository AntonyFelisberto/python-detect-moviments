import numpy as np
import cv2
import sys
from random import randint
import csv

TEXT_COLOR = (randint(0,255),randint(0,255),randint(0,255))
BORDER_COLOR = (randint(0,255),randint(0,255),randint(0,255))

FONT = cv2.FONT_HERSHEY_COMPLEX
TEXT_SIZE = 1.2
TITLE_TEXT_POSITION = (100,40)

VIDEO_SOURCE = "arquivos\\cars\\PEOPLE.mp4"

BGS_TYPES = ["GMG","MOG2","MOG","KNN","CNT"]
BGS_TYPE = BGS_TYPES[4]

def get_bg_subtractor(BGS_TYPE):
    if BGS_TYPE == "GMG":
        return cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=120,decisionThreshold=0.8)
    if BGS_TYPE == "MOG":
        return cv2.bgsegm.createBackgroundSubtractorMOG(history=200,nmixtures=5,backgroundRatio=0.7,noiseSigma=0)
    if BGS_TYPE == "MOG2":
        return cv2.createBackgroundSubtractorMOG2(history=500,detectShadows=True,varThreshold=100)
    if BGS_TYPE == "KNN":
        return cv2.createBackgroundSubtractorKNN(history=500,dist2Threshold=400,detectShadows=True)
    if BGS_TYPE == "CNT":
        return cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15,useHistory=True,maxPixelStability=15*60,isParallel=True)
    
    print("detector invalido")
    sys.exit(1)

cap = cv2.VideoCapture(VIDEO_SOURCE)
bg_subtractor = get_bg_subtractor(BGS_TYPE)
ciclo_inicial = cv2.getTickCount()

def main():
    frame_number = 1
    while cap.isOpened():
        ok, frame = cap.read()

        if not ok:
            print("erro")
            break

        frame_number += 1

        bg_mask = bg_subtractor.apply(frame)
        res = cv2.bitwise_and(frame,frame,mask=bg_mask)

        cv2.imshow("original", frame)
        cv2.imshow("mask", res)

        if cv2.waitKey(1) & 0xff == ord("q") or frame_number > 250: # frame_number é para não fazer o video inteiro
            break

    ciclo_final = cv2.getTickCount()
    t = (ciclo_final- ciclo_inicial) / cv2.getTickFrequency()
    print(t)
main()