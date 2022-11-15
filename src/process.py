import cv2 as cv

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--tracker_output', default="output.mov")
    args = parser.parse_args()

    cap = cv.VideoCapture(args.input)

    feature_collection = []

    cv.namedWindow('tracked', flags=cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    cv.resizeWindow('tracked', 500, 500)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        orb = cv.ORB_create()
        kp = orb.detect(frame, None)

        kp, desc = orb.compute(frame, kp)

        img2 = cv.drawKeypoints(frame, kp, None, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv.imshow('tracked', img2)
        if cv.waitKey(100) == 27: # escape
            break
    