# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (480, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(480, 480))

# allow the camera to warmup
time.sleep(0.1)

counter = 0

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array
    #image_small = cv2.resize(image, (240, 240))
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.rectangle(image_gray, (30, 10), (130, 110), (255, 255, 255), 3)

    # face reconition
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)

    for (x,y,w,h) in faces:
        image_face = image_gray[y:y+h, x:x+w]
        image_edit = cv2.resize(image_face, (240, 240))
        cv2.rectangle(image_gray,(x,y),(x+w,y+h),(255,255,255),2)

        # capture a photo
        cv2.imwrite('./tmp/pic' + str(counter) + '.jpg', image_edit)
        print("current count: " + str(counter))
        counter += 1

    # show the frame
    cv2.imshow("Frame", image_gray)
    key = cv2.waitKey(1) & 0xFF



    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
