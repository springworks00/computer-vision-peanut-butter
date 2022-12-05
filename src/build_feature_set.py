from email.policy import default
import json
import cv2 as cv
import numpy as np
from operator import itemgetter

import math
import argparse
import feature

def dump_features(features_collection, file : str) -> None:
    with open(file, 'w') as f:
        f.write(json.dumps(features_collection, cls=feature.FeatureJSONEncoder))

def load_features(file : str) -> list[feature.Feature]:
    with open(file, 'r') as f:
        return json.loads(f.read(), cls=feature.FeatureJSONDecoder)

def rotation_from_H(H):
    u, _, vh = np.linalg.svd(H[0:2, 0:2])
    R = u @ vh
    return R

def theta_from_R(R):
    return math.atan2(R[1,0], R[0,0])

def estimate_essential(pts1, pts2):
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)

    if not F is None:
        # select only inlier points
        pts1 = pts1[mask.ravel()==1]
        pts2 = pts2[mask.ravel()==1]

        u, _, vh = np.linalg.svd(F)

        print(F)
        print("MARK")

        E, mask = cv.findEssentialMat(pts1, pts2)

def make_feature(feature_builder, frame : cv.UMat) -> feature.Feature:
    new_fr_kp = feature_builder.detect(frame, None)
    new_fr_kp, new_fr_desc = feature_builder.compute(frame, new_fr_kp)

    return feature.Feature(new_fr_kp, new_fr_desc, frame)

class FeatureExtractor:
    def __init__(self, feature_frames : int, matcher, feature_builder) -> None:
        self.feature_builder = feature_builder

        self.feature_collection = []
        self.feature_votes = []

        self.feature_frames = feature_frames
        self.matcher = matcher

    def add_feature_frame(self, frame : cv.UMat) -> None:
        self.feature_collection.append(make_feature(self.feature_builder, frame))

    def reduce_features(self):
        for i, feature_fr in enumerate(self.feature_collection):
            votes = np.zeros((feature_fr.descriptors.shape[0]))
            a = max(0, i-self.feature_frames)
            b = min(len(self.feature_collection), i+self.feature_frames)
            for j in range(a, b):
                if j == i:
                    continue
                fr = self.feature_collection[j]
                # feature1 : feature.Feature = feature_collection[-1]
                matches = self.matcher.knnMatch(np.float32(feature_fr.descriptors), np.float32(fr.descriptors), k=2)

                # store all the good matches as per Lowe's ratio test.
                good = []
                for m,n in matches:
                    if m.distance < 0.7*n.distance:
                        good.append(m)

                if len(good) < 4:
                    continue

                for match in good:
                    votes[match.queryIdx] += 1

                n_fr_pts, fr_pts = map(list, zip(*[(np.array(feature_fr.keypoints[match.queryIdx].pt)[np.newaxis, :], np.array(fr.keypoints[match.trainIdx].pt)[np.newaxis, :]) for match in good]))
                n_fr_pts = np.stack(n_fr_pts, axis=0)
                fr_pts = np.stack(fr_pts, axis=0)

                # H, mask = cv.findHomography(n_fr_pts, fr_pts, method=cv.RANSAC, ransacReprojThreshold=5.0)
                # mask = mask.ravel().tolist()

                

                draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                                singlePointColor = None,
                                # matchesMask = mask, # draw only inliers
                                flags = 2)

                im3 = cv.drawMatches(feature_fr.frame, feature_fr.keypoints, fr.frame, fr.keypoints, good, None, **draw_params)

                cv.imshow("tracked", im3)
                if cv.waitKey(1) == 27: # escape
                    stop = True
                    break
            
            self.feature_votes.append(votes)

        assert(len(self.feature_votes) == len(self.feature_collection))

        total_features = 0
        final_features = 0
        for f in self.feature_collection:
            total_features += len(f.descriptors)

        print("Reducing feature count")
        for f, vote in zip(self.feature_collection, self.feature_votes):
            f.descriptors = f.descriptors[vote > 0]
            f.keypoints = [x for x, y, in zip(f.keypoints, vote) if y > 0]

        for f in self.feature_collection:
            final_features += len(f.descriptors)

        print("%i features down from %i" % (final_features, total_features))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--tracker_output', default="output.mov")
    parser.add_argument('--feature_file', required=False, default='features.json')
    parser.add_argument('--feature_input')
    parser.add_argument('--frame_skip', default=30, type=int)
    parser.add_argument('--feature_frames', default=4, type=int)
    args = parser.parse_args()

    cap = cv.VideoCapture(args.input)

    FLANN_INDEX_KDTREE = 0
    flann_params = dict(algorithm = FLANN_INDEX_KDTREE,
                        trees = 4)

    flann_matcher = cv.FlannBasedMatcher(flann_params)

    sift = cv.SIFT_create()

    fextractor = FeatureExtractor(args.feature_frames, cv.BFMatcher_create(), sift)

    cv.namedWindow('tracked', flags=cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    cv.resizeWindow('tracked', 500, 500)

    frame_id = 0
    stop = False
    while cap.isOpened() and not stop:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_id % args.frame_skip) == 0:
            fextractor.add_feature_frame(frame)

        frame_id += 1

    # fextractor.reduce_features()

    dump_features(fextractor.feature_collection, args.feature_file)