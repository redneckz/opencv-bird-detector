import numpy as np
from time import time

DT_EPS = 1 / 1000
DEFAULT_FPS = 30
WINDOW_SIZE = DEFAULT_FPS * 10 # It is about half a minute at 60fps

class FPSProfiler:
    def __init__(self):
        self.measurements = np.empty(WINDOW_SIZE)
        self.measurements.fill(DEFAULT_FPS)
        self.pointer = 0

    def __iter__(self):
        yield self.measurements[self.pointer - 1]
        yield self.measurements.min()
        yield self.measurements.mean()
        yield self.measurements.max()

    def profile(self, func):
        startTime = time()
        result = func()
        deltaTime = time() - startTime + DT_EPS
        currentFPS = 1 / deltaTime
        self.measurements[self.pointer] = currentFPS
        self.pointer = (self.pointer + 1) % WINDOW_SIZE
        return result
