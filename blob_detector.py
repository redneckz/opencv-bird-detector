import cv2
import numpy as np
from sklearn.cluster import dbscan

class BlobDetector:
    def prepare(self):
        params = cv2.SimpleBlobDetector_Params()
        # Source image is binary. So no thresholding is required
        params.minThreshold = 0
        params.maxThreshold = 0xFF
        params.thresholdStep = params.maxThreshold - params.minThreshold;
        params.minRepeatability = 0
        # Blobs merging by distance
        self.minBlobDistance = 150
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 2**2
        params.maxArea = 1000**2
        # Filter by Circularity
        params.filterByCircularity = False
        params.minCircularity = 0.8
        params.maxCircularity = 1
        # Filter by Convexity
        params.filterByConvexity = False
        params.minConvexity = 0.95
        params.maxConvexity = 1
        # Filter by Inertia
        params.filterByInertia = False
        params.minInertiaRatio = 0.1
        params.maxInertiaRatio = 1
        self.blobDetector = cv2.SimpleBlobDetector_create(params)

    def detect(self, frame):
        invertedFrame = cv2.bitwise_not(frame)
        keypoints = self.blobDetector.detect(invertedFrame)
        return mergeKeypoints(keypoints, self.minBlobDistance)

def mergeKeypoints(keypoints, eps):
    if len(keypoints) < 2:
        return keypoints
    points = np.array([keypoint.pt for keypoint in keypoints])
    sizes = np.array([keypoint.size for keypoint in keypoints])
    # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.dbscan.html
    _, pointsLabels = dbscan(points, eps = eps, min_samples = 1, metric = 'euclidean')
    clustersPredicates = ((pointsLabels == label) for label in range(0, max(pointsLabels) + 1))
    mergedKeypoints = [createCentroid(points[predicate], sizes[predicate], eps) for predicate in clustersPredicates]
    return mergedKeypoints

def createCentroid(clusterPoints, clusterSizes, eps):
    centroidPoint = np.mean(clusterPoints, axis = 0)
    centroidSize = np.clip(np.sum(clusterSizes), eps // 4, 2 * eps)
    return cv2.KeyPoint(centroidPoint[0], centroidPoint[1], centroidSize)
