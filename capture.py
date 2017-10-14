import cv2
import numpy as np
from matplotlib import pyplot as plt

capture = cv2.VideoCapture(0)
if not capture.isOpened():
    quit()

while not capture.grab():
    cv2.waitKey(100)

ok, frame = capture.retrieve()
while capture.grab():
    grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, 1)
    edges = cv2.Canny(grayImg, 100, 200)
    cv2.imshow('Capture', edges)
    cv2.waitKey(100)
    ok, frame = capture.retrieve()

cv2.waitKey(0)
capture.release()
cv2.destroyAllWindows()
