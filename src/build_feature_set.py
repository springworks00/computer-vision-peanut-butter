from email.policy import default
import json
import cv2 as cv
import numpy as np

import argparse
import feature

def dump_features(features, file : str) -> None:
    with open(file, 'w') as f:
        f.write(json.dumps(features, cls=feature.FeatureJSONEncoder))

def load_features(file : str) -> list[feature.Feature]:
    with open(file, 'r') as f:
        return json.loads(f.read(), cls=feature.FeatureJSONDecoder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--tracker_output', default="output.mov")
    parser.add_argument('--feature_file', required=False, default='features.json')
    parser.add_argument('--frame_skip', default=4, type=int)
    parser.add_argument('--feature_frames', default=4, type=int)
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

            new_feature_fr = feature.Feature(kp, desc, frame)

            img2 = cv.drawKeypoints(frame, kp, None, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # cv.imshow('tracked', img2)
            
            if len(feature_collection) > args.feature_frames:
                for i, fr in enumerate(feature_collection[-args.feature_frames:]):
                    # feature1 : feature.Feature = feature_collection[-1]
                    matches = flann_matcher.knnMatch(np.float32(new_feature_fr.descriptors), np.float32(fr.descriptors), 2)

                    # store all the good matches as per Lowe's ratio test.
                    good = []
                    for m,n in matches:
                        if m.distance < 0.7*n.distance:
                            good.append(m)

                    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                                    singlePointColor = None,
                                    #    matchesMask = matchesMask, # draw only inliers
                                    flags = 2)

                    im3 = cv.drawMatches(new_feature_fr.frame, new_feature_fr.keypoints, fr.frame, fr.keypoints, good, None, **draw_params)
                    cv.imshow("tracked", im3)
                    if cv.waitKey(100) == 27: # escape
                        exit(-1)
                if True: # TODO evaluate frame quality
                    feature_collection.append(new_feature_fr)
            else:
                feature_collection.append(new_feature_fr)

        frame_id += 1

    dump_features(feature_collection, args.feature_file)