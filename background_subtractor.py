import cv2

DEFAULT_KERNEL_SIZE = 3

class BackgroundSubtractor:
    def prepare(self):
        # https://docs.opencv.org/3.0-last-rst/doc/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html#py-background-subtraction
        self.morphKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DEFAULT_KERNEL_SIZE, DEFAULT_KERNEL_SIZE))
        self.backgroundSubtractor = cv2.bgsegm.createBackgroundSubtractorGMG()

    def apply(self, frame):
        foregroundMask = self.backgroundSubtractor.apply(frame)
        denoisedMask = cv2.morphologyEx(foregroundMask, cv2.MORPH_OPEN, self.morphKernel)
        return denoisedMask
