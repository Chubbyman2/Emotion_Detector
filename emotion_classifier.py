import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image

# Use Haar Cascade algorithm to detect faces
# and return coordinates of the rectangle around the face
face_classifier = cv2.CascadeClassifier(
    "C:/Users/charl/OneDrive/Desktop/Machine Learning Stuff/haarcascade_frontalface_default.xml")
classifier = load_model(
    "C:/Users/charl/OneDrive/Desktop/Machine Learning Stuff/Emotion_Detector.h5")

classes = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Convert to binary colour image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Store coordinates of rectangle in list, faces
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Blue bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Stores only face from the image
        roi_gray = gray[y:y+h, x:x+w]

        # Resize to 224x224 using interpolation to calculate pixel values for new image
        roi_gray = cv2.resize(roi_gray, (48, 48),
                              interpolation=cv2.INTER_AREA)

        # If a face is detected in the ROI by the classifier
        if np.sum([roi_gray]) != 0:

            # Normalized, converted to array to be used by model
            roi = roi_gray.astype("float")/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            predict = classifier.predict(roi)[0]
            label = classes[predict.argmax()]
            label_position = (x, y)

            # Puts prediction as text above the ROI, in green
            cv2.putText(frame, label, label_position,
                        cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "No Face Detected", (20, 20),
                        cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow("Emotion Detector", frame)

    # Press Esc to quit
    k = cv2.waitKey(25) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
