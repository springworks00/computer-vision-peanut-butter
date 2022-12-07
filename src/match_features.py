import json
import cv2
import numpy as np
import argparse
import copy
import build_feature_set
import random
import sys
import math

def get_frame_features(sift, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_kp, frame_des = sift.detectAndCompute(frame, None)
    return frame_kp, frame_des

# video_path: path to video file
# calc_fps: how many frames per second to calculate features
def calculate_video_features(video_path, calc_fps=2):
    # Foreground detection?
    print(f"Video Path: {video_path}")
    cap = cv2.VideoCapture(video_path)
    calc_every_n_frames = int(cap.get(cv2.CAP_PROP_FPS)/calc_fps)

    # SIFT is giving best quality matches
    video_sift = cv2.SIFT_create()
    
    if (cap.isOpened()== False):
        print("Error opening video file")
    
    quality_features = [] # list of quality features we should be matching against

    frame_count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            if frame_count == calc_every_n_frames:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_kp, frame_des = video_sift.detectAndCompute(frame, None)
                show_img = cv2.drawKeypoints(frame, frame_kp, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                quality_features.append((frame, show_img, frame_kp, frame_des))
                frame_count = 0
            else:
                show_img = frame

            frame_count += 1
            cv2.imshow('Frame', show_img)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            break

    # for frame, frame_kp_drawn, frame_kp, frame_des in quality_features:
    #     cv2.imshow('Quality Frame', frame_kp_drawn)
    #     if cv2.waitKey(50) & 0xFF == ord('q'):
    #         break

    cap.release()
    cv2.destroyAllWindows()
    quality_features = [(frame, frame_kp, frame_des) for frame, frame_kp_drawn, frame_kp, frame_des in quality_features] # dont need drawn kps anymore
    return quality_features

def calculate_img_features(img_path, nfeatures=100000):
    print(f"Image Path: {img_path}")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # fast score giving more features around edges, don't know how different from default
    # TODO: Could we use sift for features on image and then match against orb features in the video?
    img_sift = cv2.SIFT_create()

    img_kp, img_des = img_sift.detectAndCompute(img, None)
    show_img = cv2.drawKeypoints(img, img_kp, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('Image', show_img)
    cv2.waitKey() & 0xFF == ord('q')

    return img, img_kp, img_des

def homography_reprojection_err(M, src_pts, dest_pts):
    reprojected_pts = cv2.perspectiveTransform(src_pts, M).reshape(-1, 2)
    dest_pts = dest_pts.reshape(-1, 2)
    return np.linalg.norm(reprojected_pts - dest_pts)

def compute_possible_homographies(video_features, img, img_kp, img_des, ratio_test=0.75, good_match_thres=10):
    # NOTE: FLANN or BF? Choosing BFMatcher here because it is guaranteed to find the nearest neighbors
    # FLANN is fast, not as accurate

    bf = cv2.BFMatcher() # norm hamming required for orb features
    img_clean = copy.deepcopy(img) # clean image for final homography display

    # Computing homography via RANSAC
    poss_homographies = [] # [(reprojection_err, homography, matches, matches_mask, frame, frame_kp, frame_des), ...]
    for frame, frame_kp, frame_des in video_features:
        matches = bf.knnMatch(frame_des, img_des, k=2)

        # NOTE: If don't want to do ratio test, can use cross check on BFMatcher (it is an alternative)
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_test * n.distance:
                good_matches.append(m)

        if len(good_matches) > good_match_thres:
            print("Possible Object!")
            frame_pts = np.float32([ frame_kp[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
            img_pts = np.float32([ img_kp[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(frame_pts, img_pts, cv2.RANSAC, 5.0)
            M_err = homography_reprojection_err(M, frame_pts, img_pts)
            good_matches_mask = mask.ravel().tolist()
            poss_homographies.append((M_err, M, good_matches, good_matches_mask, frame, frame_kp, frame_des))

            h,w = frame.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

            img_w_bound_box = cv2.polylines(img, [np.int32(dst)], True, 255,3, cv2.LINE_AA)

            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                singlePointColor = None,
                matchesMask = good_matches_mask, # draw only inliers
                flags = 2)
            show_img = cv2.drawMatches(frame, frame_kp, img_w_bound_box, img_kp, good_matches, None, **draw_params)
        else:
            print("Object Not Found!")
            show_img = cv2.drawMatches(frame, frame_kp, img, img_kp, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv2.imshow('Matches', show_img)
        if cv2.waitKey(300) & 0xFF == ord('q'):
            break

    # Get best homography and draw it
    sorted_homographes = sorted(poss_homographies, key=lambda x: (len(x[2]) * -1, x[0]))
    print([(x[0], len(x[2])) for x in sorted_homographes])
    M_best_err, M_best, M_best_matches, M_best_matches_mask, M_best_frame, M_best_frame_kp, M_best_frame_des =  sorted_homographes[0]
    print(M_best_err)
    h,w = M_best_frame.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M_best)
    img_w_bound_box = cv2.polylines(img_clean, [np.int32(dst)], True, 255,3, cv2.LINE_AA)
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
        singlePointColor = None,
        matchesMask = M_best_matches_mask, # draw only inliers
        flags = 2)
    show_img = cv2.drawMatches(M_best_frame, M_best_frame_kp, img_w_bound_box, img_kp, M_best_matches, None, **draw_params)
    cv2.imshow('Best Homography', show_img)
    cv2.waitKey() & 0xFF == ord('q')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_json', default="features.json")
    parser.add_argument('--video', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--feature_frames', default=4, type=int)
    parser.add_argument('--frame_skip', default=30, type=int)
    args = parser.parse_args()

    print(f"OpenCV Version: {cv2.__version__}")

    video_path = args.video
    img_path = args.image

    # -------

    # TODO: Implement multiprocessing to speed things up a ton

    # NOTE: DO WE NEED TO CONVERT TO GRAYSCALE FOR ORB FEATURES TO WORK?

    # STEP 1: Compile set of "quality" features from the video
        # - Default ORB parameters for video should be good (oriented at center of image, fast)
    # video_features = calculate_video_features(video_path) # video_features [(frame, frame_kp, frame_des), ...]

    # INTEGRATION
    # cap = cv2.VideoCapture(args.video)

    # FLANN_INDEX_KDTREE = 0
    # flann_params = dict(algorithm = FLANN_INDEX_KDTREE,
    #                     trees = 4)

    # flann_matcher = cv2.FlannBasedMatcher(flann_params)

    # sift = cv2.SIFT_create()

    # fextractor = build_feature_set.FeatureExtractor(args.feature_frames, cv2.BFMatcher_create(), sift)

    # cv2.namedWindow('tracked', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    # cv2.resizeWindow('tracked', 500, 500)

    # frame_id = 0
    # stop = False
    # while cap.isOpened() and not stop:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     if (frame_id % args.frame_skip) == 0:
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         fextractor.add_feature_frame(frame)

    #     frame_id += 1

    # fextractor.reduce_features()
    # video_features = [(video_feature_obj.frame, video_feature_obj.keypoints, video_feature_obj.descriptors) for video_feature_obj in fextractor.feature_collection]

    # -------

    # # STEP 2: Get features of the image
    # img, img_kp, img_des = calculate_img_features(img_path)

    # # STEP 3: Calculate the homographies present across all frames and the image
    # poss_homographies = compute_possible_homographies(video_features, img, img_kp, img_des)

    # STEP 4: Find the homography with the lowest reprojection error