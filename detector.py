import cv2
import numpy as np
cv2.namedWindow("Window")
def empty(x):
    pass
cap=cv2.VideoCapture(0)

success,frame1=cap.read()
#frame1=cv2.resize(frame1,(0,0),None,0.2,0.2)
success,frame2=cap.read()
#frame2=cv2.resize(frame2,(0,0),None,0.2,0.2)
cv2.createTrackbar("Threshold1","Window",20,255,empty)
cv2.createTrackbar("Threshold2","Window",255,255,empty)
while True:
    
    diff=cv2.absdiff(frame2,frame1)
    diffgray=cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    diffblur=cv2.GaussianBlur(diffgray,(5,5),0)
    thresh1=cv2.getTrackbarPos("Threshold1","Window")
    thresh2=cv2.getTrackbarPos("Threshold2","Window")
    _,thr=cv2.threshold(diffblur,thresh1,thresh2,cv2.THRESH_BINARY)
    
    kernel=np.ones((5,5))
    imgdilate=cv2.dilate(thr,kernel,iterations=3)
    imgerode=cv2.erode(imgdilate,kernel,iterations=1)
    #cv2.imshow("imgdil",imgdilate)
    #cv2.imshow("imgerr",imgerode)
    contours,_=cv2.findContours(imgerode,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area>2000:
            #cv2.drawContours(frame1,cnt,-1,(0,255,0),3)
            peri=cv2.arcLength(cnt,True)
            approx=cv2.approxPolyDP(cnt,0.02*peri,True)
            x,y,w,h=cv2.boundingRect(approx)
            cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(frame1,"Movement",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            
    cv2.imshow("track",frame1)
   
    frame1=frame2
    success,frame2=cap.read()
    #frame2=cv2.resize(frame2,(0,0),None,0.2,0.2)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
