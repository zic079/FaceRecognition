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
    image_small = cv2.resize(image, (240, 240))
    image_gray = cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY)
    #cv2.rectangle(image_gray, (30, 10), (130, 110), (255, 255, 255), 3)

    # show the frame
    cv2.imshow("Frame", image_gray)
    key = cv2.waitKey(1) & 0xFF

    # capture a photo
    #cv2.imwrite('./tmp/pic' + counter + '.jpg', image)
    #counter += 1

    print("current count: " + str(counter))
    counter += 1

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
