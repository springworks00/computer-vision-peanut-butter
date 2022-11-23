import cv2 as cv

class Feature:

    def __init__(self, frame : cv.UMat, kp, desc) -> None:
        self.frame = frame
        self.keypoints = kp
        self.descriptors = desc