import cv2
from capture_camera import captureCamera
from blob_detector import BlobDetector

blobDetector = BlobDetector()

def onStart():
    blobDetector.prepare()
    print('Detection started...')

def onFrame(frame):
    cv2.imshow('Birds', blobDetector.detect(frame))

def onEnd():
    print('Detection ended.')

captureCamera(onStart, onFrame, onEnd)

cv2.destroyAllWindows()
