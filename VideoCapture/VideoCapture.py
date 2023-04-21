import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.utils import img_to_array
#load model
cnn_model = model_from_json(open("cnn/fer.json", "r").read())
#load weights
cnn_model.load_weights('cnn/fer.h5')

#load model
resnet_model = model_from_json(open("resnet/fer.json", "r").read())
#load weights
resnet_model.load_weights('resnet/resnet1.h5')


face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap=cv2.VideoCapture(0)

while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    rgb_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)


    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray = gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
        roi_rgb = rgb_img[y:y+w,x:x+h]

        roi_gray=cv2.resize(roi_gray,(48,48))
        roi_rgb = cv2.resize(roi_rgb, (48,48))

        img_pixels = img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)

        rgb_pixels = img_to_array(roi_rgb)
        rgb_pixels = np.expand_dims(rgb_pixels, axis = 0)

        img_pixels /= 255

        cnn_predictions = cnn_model.predict(img_pixels)
        resnet_predictions = resnet_model.predict(rgb_pixels)

        #find max indexed array
        cnn_index = np.argmax(cnn_predictions[0])
        resnet_index = np.argmax(resnet_predictions[0])
        max_index = np.argmax(cnn_predictions[0]) if np.max(cnn_predictions) > np.max(resnet_predictions) else np.argmax(resnet_predictions[0])
        model_choice = "CNN" if np.max(cnn_predictions) > np.max(resnet_predictions) else "ResNet"

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        cnn_predicted_emotion = emotions[cnn_index]
        resnet_predicted_emotion = emotions[resnet_index]
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(test_img, model_choice, (int(x), int(y+h+10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(test_img, f"CNN {cnn_predicted_emotion}: {int(cnn_predictions[0][max_index] * 10000)/100}%", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(test_img, f"ResNet {resnet_predicted_emotion}: {int(resnet_predictions[0][max_index]*10000)/100}%", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ',resized_img)

    # break

    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows