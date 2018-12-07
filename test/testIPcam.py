import numpy as np
import cv2

# cap = cv2.VideoCapture('http://ip/video.mjpg')
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('http://140.113.73.205:8081/')

while True:
    # cap = cv2.VideoCapture('http://ip/out.jpg')
    ret, frame = cap.read()
    print(ret)
    cv2.imshow('image', frame)
    cv2.waitKey(1)
