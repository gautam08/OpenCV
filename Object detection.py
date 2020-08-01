import cv2
import numpy as np
cap=cv2.VideoCapture(0)

while(True):
    ret,frame=cap.read()
    frame=cv2.flip(frame,+1)
    frame2=cv2.cvtColor(frame,cv2.COLOR_BGR2HLS)
    lb=np.array([0,0,81])
    ub=np.array([185,255,255])
    mask=cv2.inRange(frame2,lb,ub)
    cv2.imshow('mask',mask)
    kernel = np.ones((5,5),np.uint8)
    opening=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    cv2.imshow('opening',opening)
    contours,hierarchy=cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    x1=[]
    y1=[]
    for contour in contours:
        if cv2.contourArea(contour)>1500:
            cnt=contour
            M=cv2.moments(cnt)
            cx=int(M['m10'] / M['m00'])
            cy=int(M['m10'] / M['m00'])
            cv2.circle(frame,(cx,cy),5,[50,120,255],-1)
            extTop=tuple(cnt[cnt[:, :, 1].argmin()][0])

            if abs(cy - extTop[1]) > 200 and abs(cx - extTop[0])< 150:
                x1.append(extTop[0])
                y1.append(extTop[1])
            for i in range(len(x1)):
                cv2.circle(frame,(x1[i],y1[i]), 4 , (255,155,100), 5)
    cv2.imshow('Result Image',frame)
    if  cv2.waitKey(20) & 0xff==ord('a'):
        break
cv2.destroyAllWindows()
cap.release()
