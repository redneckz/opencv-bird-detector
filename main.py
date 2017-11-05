import cv2
import numpy as np
from capture_camera import captureCamera
from frame_filter import FrameFilter
from background_subtractor import BackgroundSubtractor
from blob_detector import BlobDetector
from bird_detector import BirdDetector
from fps_profiler import FPSProfiler

# Use numeric keys to switch between algorithm stages
STAGES_TITLES = ('SOURCE FRAME', 'FILTERED FRAME', 'FOREGROUND MASK', 'BLOBS FRAME')

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SIZE = 0.4
ACTIVE_STAGE_INDEX_COLOR = (0xFF, 0xFF, 0xFF)
FPS_COLOR = (0xFF, 0xFF, 0xFF)

NUM_1_KEY = 49

class DetectorRunner:
    def __init__(self):
        self.birdDetector = BirdDetector(
            frameFilter = FrameFilter(),
            backgroundSubtractor = BackgroundSubtractor(),
            blobDetector = BlobDetector()
        )
        self.activeStageIndex = -1 # By default show last stage
        self.profiler = FPSProfiler()

    def onStart(self):
        self.birdDetector.prepare()
        print('Detection started...')

    def onFrame(self, frame):
        self.stages = self.profiler.profile(lambda: self.birdDetector.detect(frame))
        activeStage = self.stages[self.activeStageIndex]
        self.drawActiveStageIndex(activeStage)
        self.drawFPS(activeStage)
        cv2.imshow('Detector', activeStage)

    def drawActiveStageIndex(self, activeStage):
        stageCount = len(self.stages)
        stageIndex = (stageCount + self.activeStageIndex) % stageCount
        cv2.putText(
            activeStage,
            'STAGE #%s <%s>' % (stageIndex + 1, STAGES_TITLES[stageIndex]),
            (0, 10), FONT, FONT_SIZE, ACTIVE_STAGE_INDEX_COLOR
        )

    def drawFPS(self, activeStage):
        cv2.putText(
            activeStage,
            'FPS%5.1f MIN%5.1f AVG%5.1f MAX%5.1f' % tuple(self.profiler),
            (0, 21), FONT, FONT_SIZE, FPS_COLOR
        )

    def onEnd(self):
        print('Detection ended.')

    def onKey(self, key):
        stageCount = len(self.stages)
        if 0 <= (key - NUM_1_KEY) < stageCount:
            self.activeStageIndex = np.clip(key - NUM_1_KEY, 0, stageCount - 1)

    def run(self):
        captureCamera(
            self.onStart,
            self.onFrame,
            self.onEnd,
            self.onKey
        )
        cv2.destroyAllWindows()

detectorRunner = DetectorRunner()
detectorRunner.run()
