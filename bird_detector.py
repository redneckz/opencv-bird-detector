import cv2
import numpy as np

KEYPOINTS_COLOR = (0, 0, 0xFF)

class BirdDetector:
    def __init__(self, frameFilter, backgroundSubtractor, blobDetector):
        self.frameFilter = frameFilter
        self.backgroundSubtractor = backgroundSubtractor
        self.blobDetector = blobDetector

    def prepare(self):
        self.frameFilter.prepare()
        self.backgroundSubtractor.prepare()
        self.blobDetector.prepare()

    def detect(self, frame):
        denoisedFrame = self.frameFilter.filter(frame)
        foregroundMask = self.backgroundSubtractor.apply(denoisedFrame)
        keypoints = self.blobDetector.detect(foregroundMask)
        blobsFrame = cv2.drawKeypoints(
            denoisedFrame,
            keypoints,
            np.array([]),
            KEYPOINTS_COLOR,
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        return (frame, denoisedFrame, foregroundMask, blobsFrame)
