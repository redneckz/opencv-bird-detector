import cv2

class BlobDetector:
    def prepare(self):
        self.backgroundSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

    def detect(self, frame):
        grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, 1)
        denoisedImg = cv2.bilateralFilter(grayImg, 5, 75, 75)
        foregroundImg = self.backgroundSubtractor.apply(denoisedImg)
        return foregroundImg
