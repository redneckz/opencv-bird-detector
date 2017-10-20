import cv2

def waitForEscape(delay = 100):
    return cv2.waitKey(delay) == 27
