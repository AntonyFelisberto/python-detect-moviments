import numpy as np
import cv2

VIDEO_SOURCE = "arquivos\\cars\\cars.mp4"
VIDEO_OUT = "filtragem_mediana_temporal.avi"

cap = cv2.VideoCapture(VIDEO_SOURCE)
has_frame,frame = cap.read()
print(has_frame,frame.shape)

fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter(VIDEO_OUT,fourcc,25,(frame.shape[1],frame.shape[0]),False)

print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(np.random.uniform(size=25))
frame_ids = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)
print(frame_ids)

frames = []
for fid in frame_ids:
    cap.set(cv2.CAP_PROP_POS_FRAMES,fid)
    has_frames, frame = cap.read()
    frames.append(frame)

print(np.mean([1,2,3,5,6,7,8,9]))
print(np.median([1,2,3,5,6,7,8,9]))

print(np.median(frames, axis=0).astype(dtype=np.uint8))

median_frames = np.median(frames,axis=0).astype(dtype=np.uint8)
print(median_frames)
cv2.waitKey(0)

cap.set(cv2.CAP_PROP_POS_FRAMES,0)
gray_median_frame = cv2.cvtColor(median_frames,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray",gray_median_frame)
cv2.waitKey(0)

while True:
    has_frame, frame = cap.read()

    if not has_frame:
        print("erros")
        break

    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    dframe = cv2.absdiff(frame_gray,gray_median_frame)

    th, dframe = cv2.threshold(dframe,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print(th)
    writer.write(dframe)

    cv2.imshow("Frames",frame_gray)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

writer.release()
cap.release()