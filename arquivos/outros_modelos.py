import numpy as np
import cv2
import sys
from random import randint

TEXT_COLOR = (randint(0,255),randint(0,255),randint(0,255))
BORDER_COLOR = (randint(0,255),randint(0,255),randint(0,255))
FONT = cv2.FONT_HERSHEY_COMPLEX
VIDEO_SOURCE = "arquivos\\cars\\cars.mp4"

BGS_TYPES = ["GMG","MOG2","MOG","KNN","CNT"]

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

def main():
    while cap.isOpened():
        ok, frame = cap.read()

        if not ok:
            print("erro")
            break

        frame = cv2.resize(frame, (0,0),fx=0.5,fy=0.5)

        if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
            print(frame.shape)

            bg_mask = bg_subtractor.apply(frame)

            fg_mask = get_filter(bg_mask,"dilation")
            fg_mask_closing = get_filter(bg_mask,"opening")
            fg_mask_opening = get_filter(bg_mask,"clossing")
            fg_mask_combine = get_filter(bg_mask,"combine")

            res = cv2.bitwise_and(frame,frame,mask=fg_mask)
            res_closing = cv2.bitwise_and(frame,frame,mask=fg_mask_closing)
            res_opening = cv2.bitwise_and(frame,frame,mask=fg_mask_opening)
            res_combine = cv2.bitwise_and(frame,frame,mask=fg_mask_combine)

            res_sem_processamento = cv2.bitwise_and(frame,frame,mask=bg_mask)

            cv2.putText(res,"background subtract"+ BGS_TYPE,(10,50),FONT,1,BORDER_COLOR,3,cv2.LINE_AA)
            cv2.putText(res,"background subtract"+ BGS_TYPE,(10,50),FONT,1,TEXT_COLOR,2,cv2.LINE_AA)



            if BGS_TYPE != "MOG" and BGS_TYPE != "GMG":
                cv2.imshow("MODEL",bg_subtractor.getBackgroundImage())

            cv2.imshow("final",res)
            cv2.imshow("res_closing",res_closing)
            cv2.imshow("res_opening",res_opening)
            cv2.imshow("res_combine",res_combine)

            if cv2.waitKey(1) == ord("q"):
                break

            def testes_mostragem():
                cv2.imshow("frame",frame)
                cv2.imshow("bg mask",bg_mask)
                cv2.imshow("bg mask filter",fg_mask)
                cv2.imshow("final",res)
                cv2.imshow("sem pre procesoo",res_sem_processamento)
        else:
            print("video terminated")

cap = cv2.VideoCapture(VIDEO_SOURCE)
bg_subtractor = get_bg_subtractor(BGS_TYPES[1])
BGS_TYPE = BGS_TYPES[1]
main()