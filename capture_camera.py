import cv2
from noop import noop

ESC_KEY = 27

def captureCamera(onStart = noop, onFrame = noop, onEnd = noop, onKey = noop):
    capture = cv2.VideoCapture(0) # default camera
    onStart();
    while True:
        key = cv2.waitKey(10);
        if key == ESC_KEY: break
        if key != -1: onKey(key)
        if not capture.grab(): continue
        ok, frame = capture.retrieve()
        onFrame(frame)
    capture.release()
    onEnd()
