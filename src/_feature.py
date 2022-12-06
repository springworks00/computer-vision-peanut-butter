import cv2 as cv

class Feature:

    def __init__(self, frame : cv.UMat) -> None:
        self.frame = frame