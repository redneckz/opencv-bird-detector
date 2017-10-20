import cv2
from wait_for_escape import waitForEscape
from noop import noop

def captureCamera(onStart = noop, onFrame = noop, onEnd = noop):
    capture = cv2.VideoCapture(0) # default camera
    onStart();
    while not waitForEscape():
        if not capture.grab(): continue
        ok, frame = capture.retrieve()
        onFrame(frame)
    capture.release()
    onEnd()
