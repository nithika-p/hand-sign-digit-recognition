from keras.models import model_from_json
import sys,os
import operator
import cv2 as cv

json_file=open("model-bw.json","r")
json_model = json_file.read()
json_file.close()
loaded_model = model_from_json(json_model)

loaded_model.load_weights("model-bw.h5")

cap = cv.VideoCapture(0)
while True:

    _,frame=cap.read()
    frame = cv.flip(frame,1)
    x1 = int(0.25*frame.shape[1])
    y1=10
    x2=frame.shape[1]-10
    y2 = int(0.55*frame.shape[1])

    cv.rectangle(frame, (x1-1,y1-1), (x2+1,y2+1), (0,255,255), 1)

    roi = frame[y1:y2,x1:x2]
    roi = cv.resize(roi,(64,64))
    roi = cv.cvtColor(roi,cv.COLOR_BGR2GRAY)
    _,test_image = cv.threshold(roi,135,255,cv.THRESH_BINARY)
    cv.imshow("test",test_image)

    result = loaded_model.predict(test_image.reshape(1,64,64,1))
    prediction = {"ZERO":result[0][0],
                   "ONE":result[0][1],
                   "TWO":result[0][2],
                   "THREE":result[0][3],
                   "FOUR":result[0][4],
                   "FIVE":result[0][5]
    }
    prediction = sorted(prediction.items(), key = operator.itemgetter(1),reverse = True)
    cv.putText(frame, prediction[0][0], (10,120), cv.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)
    print(prediction)
    cv.imshow("Frame",frame)
    key = cv.waitKey(10)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()
