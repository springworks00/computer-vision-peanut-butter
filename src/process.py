import cv2 as cv
import numpy as np

import argparse
import feature

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--tracker_output', default="output.mov")
    parser.add_argument('--frame_skip', default=4, type=int)
    args = parser.parse_args()

    cap = cv.VideoCapture(args.input)

    feature_collection = []

    cv.namedWindow('tracked', flags=cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    cv.resizeWindow('tracked', 500, 500)

    FLANN_INDEX_KDTREE = 0
    flann_params = dict(algorithm = FLANN_INDEX_KDTREE,
                        trees = 4)

    flann_matcher = cv.FlannBasedMatcher(flann_params)

    frame_id = 0
    while cap.isOpened():
        if frame_id % args.frame_skip == 0:
            ret, frame = cap.read()
            if not ret:
                break

            orb = cv.ORB_create()
            kp = orb.detect(frame, None)

            kp, desc = orb.compute(frame, kp)

            feature_collection.append(feature.Feature(frame, kp, desc))

            img2 = cv.drawKeypoints(frame, kp, None, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # cv.imshow('tracked', img2)
            
            if len(feature_collection) > 2:
                feature0 : feature.Feature = feature_collection[-2]
                feature1 : feature.Feature = feature_collection[-1]
                matches = flann_matcher.knnMatch(np.float32(feature0.descriptors), np.float32(feature1.descriptors), 2)

                # store all the good matches as per Lowe's ratio test.
                good = []
                for m,n in matches:
                    if m.distance < 0.7*n.distance:
                        good.append(m)

                draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                            singlePointColor = None,
                            #    matchesMask = matchesMask, # draw only inliers
                            flags = 2)

                im3 = cv.drawMatches(feature0.frame, feature0.keypoints, feature1.frame, feature1.keypoints, good, None, **draw_params)
                cv.imshow("tracked", im3)
            if cv.waitKey(100) == 27: # escape
                break

        frame_id += 1
    