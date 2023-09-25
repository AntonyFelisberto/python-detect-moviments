import numpy as np
import cv2
import sys
from random import randint
import csv

TEXT_COLOR = (0,255,0)
TRACKER_COLOR = (255,0,0)
FONT = cv2.FONT_HERSHEY_COMPLEX
VIDEO_SOURCE = "arquivos\\cars\\PEOPLE.mp4"

BGS_TYPES = ["GMG","MOG2","MOG","KNN","CNT"]
BGS_TYPE = BGS_TYPES[0]

def get_kernel(KERNEL_TYPE):
    if KERNEL_TYPE == "dilation":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    if KERNEL_TYPE == "opening":
        return np.ones((3,3),np.uint8)
    if KERNEL_TYPE == "closing":
        return np.ones((3,3),np.uint8)

def get_filter(img,filter):
    if filter == "clossing":
        return cv2.morphologyEx(img,cv2.MORPH_CLOSE,get_kernel("clossing"),iterations=2)
    if filter == "opening":
        return cv2.morphologyEx(img,cv2.MORPH_OPEN,get_kernel("opening"),iterations=2)
    if filter == "dilation":
        return cv2.dilate(img,get_kernel("dilation"),iterations=2)
    if filter == "combine":
        closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,get_kernel("clossing"),iterations=2)
        opening = cv2.morphologyEx(closing,cv2.MORPH_OPEN,get_kernel("opening"),iterations=2)
        dilation = cv2.dilate(opening,get_kernel("dilation"),iterations=2)
        return dilation
    
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
min_area = 255
bg_subtractor = get_bg_subtractor(BGS_TYPE)

def main():
    while cap.isOpened():
        ok, frame = cap.read()

        if not ok:
            print("erro")
            break

        frame = cv2.resize(frame, (0,0),fx=0.50,fy=0.50)
        bg_mask = bg_subtractor.apply(frame)
        bg_mask = get_filter(bg_mask,'combine')
        bg_mask = cv2.medianBlur(bg_mask, 5)

        (countours, hierarchy) = cv2.findContours(bg_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in countours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(frame,(10,30),(250,55),(255,0,0),-1)
                cv2.putText(frame,"Movimento Detectado",(10,50),FONT,0.8,TEXT_COLOR,2,cv2.LINE_AA)
                cv2.drawContours(frame,cnt,-1,TRACKER_COLOR,3)
                cv2.drawContours(frame,cnt,-1,(255,255,255),1)
                cv2.rectangle(frame,(x,y),(x + w,y + h),TRACKER_COLOR,1)
                cv2.rectangle(frame,(x,y),(x + w,y + h),(255,255,255),1)

                def visualizar_alpha():
                    for alpha in np.arange(0.8,1.1,0.9)[::-1]:
                        frame_copy = frame.copy()
                        output = frame.copy()
                        cv2.drawContours(frame_copy,[cnt],-1,TRACKER_COLOR, -1)
                        frame = cv2.addWeighted(frame_copy,alpha,output,1 - alpha,0,output)

        result = cv2.bitwise_and(frame,frame,mask=bg_mask)

        cv2.imshow("frame",frame)
        cv2.imshow("result",result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

main()