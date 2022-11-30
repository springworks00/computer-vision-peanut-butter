from email.policy import default
import json
import cv2 as cv
import numpy as np
from operator import itemgetter

import math
import argparse
import feature

def dump_features(features, file : str) -> None:
    with open(file, 'w') as f:
        f.write(json.dumps(features, cls=feature.FeatureJSONEncoder))

def load_features(file : str) -> list[feature.Feature]:
    with open(file, 'r') as f:
        return json.loads(f.read(), cls=feature.FeatureJSONDecoder)

def ransac(frame0 : feature.Feature, frame1 : feature.Feature) -> tuple[cv.Mat, list[bool]]:
    pass

def rotation_from_H(H):
    u, _, vh = np.linalg.svd(H[0:2, 0:2])
    R = u @ vh
    return R

def theta_from_R(R):
    return math.atan2(R[1,0], R[0,0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--tracker_output', default="output.mov")
    parser.add_argument('--feature_file', required=False, default='features.json')
    parser.add_argument('--frame_skip', default=30, type=int)
    parser.add_argument('--feature_frames', default=4, type=int)
    args = parser.parse_args()

    cap = cv.VideoCapture(args.input)

    feature_collection = []
    feature_matches = []
    feature_votes = []

    cv.namedWindow('tracked', flags=cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    cv.resizeWindow('tracked', 500, 500)

    FLANN_INDEX_KDTREE = 0
    flann_params = dict(algorithm = FLANN_INDEX_KDTREE,
                        trees = 4)

    flann_matcher = cv.FlannBasedMatcher(flann_params)

    frame_id = 0
    stop = False
    while cap.isOpened() and not stop:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_id % args.frame_skip) == 0:

            orb = cv.ORB_create()
            new_fr_kp = orb.detect(frame, None)

            new_fr_kp, new_fr_desc = orb.compute(frame, new_fr_kp)

            new_feature_fr = feature.Feature(new_fr_kp, new_fr_desc, frame)

            img2 = cv.drawKeypoints(frame, new_fr_kp, None, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # cv.imshow('tracked', img2)
            
            if len(feature_collection) > args.feature_frames:
                votes = np.zeros((new_feature_fr.descriptors.shape[0]))
                for i, fr in enumerate(feature_collection[-args.feature_frames:]):
                    # feature1 : feature.Feature = feature_collection[-1]
                    matches = flann_matcher.knnMatch(np.float32(new_feature_fr.descriptors), np.float32(fr.descriptors), k=2)

                    # store all the good matches as per Lowe's ratio test.
                    good = []
                    for m,n in matches:
                        if m.distance < 0.7*n.distance:
                            good.append(m)

                    if len(good) < 4:
                        continue

                    for match in good:
                        votes[match.queryIdx] += 1

                    # n_fr_pts, fr_pts = map(list, zip(*[(np.array(new_feature_fr.keypoints[match.queryIdx].pt)[np.newaxis, :], np.array(fr.keypoints[match.trainIdx].pt)[np.newaxis, :]) for match in good]))
                    # n_fr_pts = np.stack(n_fr_pts, axis=0)
                    # fr_pts = np.stack(fr_pts, axis=0)

                    # H, mask = cv.findHomography(n_fr_pts, fr_pts, method=cv.RANSAC, ransacReprojThreshold=5.0)
                    # mask = mask.ravel().tolist()

                    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                                    singlePointColor = None,
                                    # matchesMask = mask, # draw only inliers
                                    flags = 2)

                    im3 = cv.drawMatches(new_feature_fr.frame, new_feature_fr.keypoints, fr.frame, fr.keypoints, good, None, **draw_params)

                    cv.imshow("tracked", im3)
                    if cv.waitKey(10) == 27: # escape
                        stop = True
                        break
                
                # new_feature_fr.keypoints = itemgetter(*[x.queryIdx for x in good])(new_feature_fr.keypoints)
                # new_feature_fr.descriptors = itemgetter(*[x.queryIdx for x in good])(new_feature_fr.descriptors)
                feature_collection.append(new_feature_fr)
                feature_votes.append(votes)
                # feature_matches.append(np.stack([np.array([m.queryIdx, m.trainIdx]) for m in good], axis=0))
            else:
                feature_collection.append(new_feature_fr)

        frame_id += 1

    # print(feature_matches)

    # for i in range(len(feature_matches)-1):
    #     matches01 = feature_matches[i]
    #     matches12 = feature_matches[i+1]
    #     feature0 = feature_collection[i]
    #     feature1 = feature_collection[i+1]
    #     feature2 = feature_collection[i+2]
    #     for m in matches01:
    #         if m.queryIdx in matches12:
    #             pass

    # print(feature_matches)

    # drop the first args.feature_frames since they were not involved in feature voting
    feature_collection = feature_collection[args.feature_frames+1:]

    total_features = 0
    final_features = 0
    for f in feature_collection:
        total_features += len(f.descriptors)

    print("Reducing feature count")
    assert(len(feature_collection) == len(feature_votes))
    for f, vote in zip(feature_collection, feature_votes):
        f.descriptors = f.descriptors[vote > 0]
        f.keypoints = [x for x, y, in zip(f.keypoints, vote) if y > 0]

    for f in feature_collection:
        final_features += len(f.descriptors)

    print("%i features down from %i" % (final_features, total_features))

    dump_features(feature_collection, args.feature_file)