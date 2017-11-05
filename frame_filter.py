import cv2

class FrameFilter:
    def prepare(self):
        pass

    def filter(self, frame):
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, 1)
        # https://docs.opencv.org/3.0-last-rst/modules/imgproc/doc/filtering.html#bilateralfilter
        # denoisedFrame = cv2.bilateralFilter(grayFrame, 5, 75, 75)
        return grayFrame
