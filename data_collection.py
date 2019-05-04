import cv2 as cv
import os
import numpy as np

if not os.path.exists("data"):
    os.makedirs("data")
    os.makedirs("data/train_data")
    os.makedirs("data/test_data")
    os.makedirs("data/train_data/0")
    os.makedirs("data/train_data/1")
    os.makedirs("data/train_data/2")
    os.makedirs("data/train_data/3")
    os.makedirs("data/train_data/4")
    os.makedirs("data/train_data/5")
    os.makedirs("data/test_data/0")
    os.makedirs("data/test_data/1")
    os.makedirs("data/test_data/2")
    os.makedirs("data/test_data/3")
    os.makedirs("data/test_data/4")
    os.makedirs("data/test_data/5")

mode='train'
directory='data/' + mode + '_data/'
cap = cv.VideoCapture(0)

while(True):
    _,frame = cap.read()
    frame = cv.flip(frame,1)
    count = {'zero':len(os.listdir(directory+"0")),
             'one':len(os.listdir(directory+"1")),
             'two':len(os.listdir(directory+"2")),
             'three':len(os.listdir(directory+"3")),
             'four':len(os.listdir(directory+"4")),
             'five':len(os.listdir(directory+"5"))}

    cv.putText(frame,"Mode : "+mode, (10,50) , cv.FONT_HERSHEY_PLAIN, 1, (255,0,255), 1 )
    cv.putText(frame,"Image Count : ", (10,100) , cv.FONT_HERSHEY_PLAIN, 1, (255,0,255), 1 )
    cv.putText(frame,"Zero : "+str(count['zero']), (10,120) , cv.FONT_HERSHEY_PLAIN, 1, (255,0,255), 1 )
    cv.putText(frame,"One : "+str(count['one']), (10,140) , cv.FONT_HERSHEY_PLAIN, 1, (255,0,255), 1 )
    cv.putText(frame,"Two : "+str(count['two']), (10,160) , cv.FONT_HERSHEY_PLAIN, 1, (255,0,255), 1 )
    cv.putText(frame,"Three : "+str(count['three']), (10,180) , cv.FONT_HERSHEY_PLAIN, 1, (255,0,255), 1 )
    cv.putText(frame,"Four : "+str(count['four']), (10,200) , cv.FONT_HERSHEY_PLAIN, 1, (255,0,255), 1 )
    cv.putText(frame,"Five : "+str(count['five']), (10,220) , cv.FONT_HERSHEY_PLAIN, 1, (255,0,255), 1 )
    #print(frame.shape)
    x1=int(0.25*frame.shape[1])
    y1=10
    x2=frame.shape[1]-10
    y2=int(0.55*frame.shape[1])
    #region of interest (roi)
    cv.rectangle(frame, (x1-1,y1-1), (x2+1,y2+1), (255,0,0),1)
    roi=frame[y1:y2, x1:x2]
    roi=cv.resize(roi, (64,64))
    cv.imshow("Frame",frame)

    roi = cv.cvtColor(roi , cv.COLOR_BGR2GRAY)
    _,roi = cv.threshold(roi,135,255,cv.THRESH_BINARY)
    cv.imshow("ROI",roi)

    key = cv.waitKey(10)
    if key == ord('0'):
        cv.imwrite(directory+'0/'+str(count['zero'])+'.jpg',roi)
    if key == ord('1'):
        cv.imwrite(directory+'1/'+str(count['one'])+'.jpg', roi)
    if key == ord('2'):
        cv.imwrite(directory+'2/'+str(count['two'])+'.jpg', roi)
    if key == ord('3'):
        cv.imwrite(directory+'3/'+str(count['three'])+'.jpg', roi)
    if key == ord('4'):
        cv.imwrite(directory+'4/'+str(count['four'])+'.jpg', roi)
    if key & 0xFF == ord('5'):
        cv.imwrite(directory+'5/'+str(count['five'])+'.jpg', roi)

cap.release()
cv.destroyAllWindows()
