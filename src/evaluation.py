import os
import cv2
import scipy.io
import numpy as np
import pandas as pd
import match_features
import tensorflow
import math
import imutils
import optical_flow

FRAMES_PATH = "data/rgbdVideoFrames"
SCENES_PATH = "data/rgbdScenes"

# gets frames paths (in order) of an object
# Ex: object_path = 'data/rgbdVideoFrames/apple/apple_1
def build_sequential_frame_paths(object_path):
    frames = []
    for frame_path in os.listdir(object_path):
        path_to_frame = object_path + "/" + frame_path
        if path_to_frame.endswith(".png"):
            frames.append(path_to_frame)
    frames = sorted(frames, key=lambda x: (int(x.split("/")[-1].split("_")[-3]), int(x.split("/")[-1].split("_")[-2])))
    return frames

def get_query_scenes(frame_skip=30):
    scene_labels = []

    for scene in os.listdir(SCENES_PATH):
        for variation in os.listdir(f"{SCENES_PATH}/{scene}"):
            if variation != ".DS_Store" and variation.endswith('.mat'):
                mat = scipy.io.loadmat(f"{SCENES_PATH}/{scene}/{variation}")
                variation = variation.replace(".mat", "")

                frame_paths = [x for x in os.listdir(f"{SCENES_PATH}/{scene}/{variation}") if not x.endswith('depth.png')]
                scene_frames_obj = [] # frame, obj_list

                count = 10
                for i in range(0, len(frame_paths)):
                    if count == 0:
                        scene_frame_path = f"{SCENES_PATH}/{scene}/{variation}/{variation}_{i+1}.png"

                        obj_paths = []
                        objects = mat['bboxes'][0][i][0]
                        for obj in objects:
                            obj_name = obj[0].flatten()[0]
                            obj_var = obj[1].flatten()[0]
                            obj_rect = obj[2].flatten()[0], obj[3].flatten()[0], obj[4].flatten()[0], obj[5].flatten()[0]
                            obj_dir = f"{FRAMES_PATH}/{obj_name}/{obj_name}_{obj_var}"
                            obj_paths.append((obj_dir, obj_rect))

                        scene_frames_obj.append((scene_frame_path, obj_paths))
                        count = frame_skip
                        count -= 1
                    else:
                        count -= 1
                
                for x in scene_frames_obj:
                    scene_labels.append(x)
    return scene_labels
     
# https://gist.github.com/IdeaKing/11cf5e146d23c5bb219ba3508cca89ec
def resize_with_pad(image, 
                    new_shape, 
                    padding_color = (0, 0, 0)):
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image

def homography_reprojection_err(M, src_pts, dest_pts):
    reprojected_pts = cv2.perspectiveTransform(src_pts, M).reshape(-1, 2)
    dest_pts = dest_pts.reshape(-1, 2)
    return np.linalg.norm(reprojected_pts - dest_pts)

def match_descriptors(bf, src_des, dest_des, ratio_test=0.75, k=2):
    matches = bf.knnMatch(src_des, dest_des ,k=k)
    good = []
    for m,n in matches:
        if m.distance < ratio_test*n.distance:
            good.append(m)
    return good

def output_stats(obj_stats):
    overall_num_present = 0
    overall_not_predicted = 0
    overall_predicted = 0
    overall_false_positive = 0
    for obj_path, correctness_dict in obj_stats.items():
        num_present = len(correctness_dict['correct']) + correctness_dict['incorrect']
        overall_num_present += num_present

        not_predicted = correctness_dict['incorrect']
        perc_not_predicted = not_predicted/num_present
        overall_not_predicted += not_predicted

        predicted = len(correctness_dict['correct'])
        perc_predicted = predicted/num_present
        overall_predicted += predicted

        fp = [x for x in correctness_dict['correct'] if x > 70]
        false_positives = len(fp)
        perc_false_positives_out_of_predicted = false_positives/predicted
        perc_false_positives_overall = false_positives/num_present
        overall_false_positive += false_positives

        print(f"Object: {obj_path}")
        print(f"Scenes Present: {num_present}")
        print(f"Predict: {predicted} | Overall Perc: {perc_predicted}")
        print(f"No Predict: {not_predicted} | Overall Perc: {perc_not_predicted}")
        print(f"False Positives: {false_positives} | Perc. of Predicted: {perc_false_positives_out_of_predicted} | Overall Perc: {perc_false_positives_overall}")
        print("-----")
    
    print(f"Total Object Presence: {overall_num_present}")
    print(f"Overall Predicted: {overall_predicted}")
    print(f"Overall Not Predicted: {overall_not_predicted}")
    print(f"Overall False Positive: {overall_false_positive}")

def run_outside_data(query_img_path, video_path):
    # sift = cv2.SIFT_create()
    # bf = cv2.BFMatcher()

    # query_img = cv2.imread(query_img_path)
    # query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    # query_kp, query_des = sift.detectAndCompute(query_img, None)

    # cap = cv2.VideoCapture(video_path)
    # while(cap.isOpened()):
    #     ret, frame = cap.read()
    #     cv2.imshow('Frame', frame)
    #     if cv2.waitKey(25) & 0xFF == ord('q'):
    #         break
    pass


if __name__ == "__main__":

    # run_outside_data("data/remote/remote_img_1.jpg", "data/remote/remote_video.MOV")

    # EVALUATION SECTION ON RGBD DATASET ----
    scene_labels = get_query_scenes()
    num_scenes = len(scene_labels)
    print(num_scenes)

    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    obj_stats = {}

    for scene_path, obj_info in scene_labels:
        print(f"Scene: {scene_path}")

        query_img = cv2.imread(scene_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
        query_kp, query_des = sift.detectAndCompute(query_img, None)

        for obj_path, obj_rect in obj_info:
            # Processing Frames of Specific Object
            obj_frame_paths = build_sequential_frame_paths(obj_path)

            # True position of the bounding box
            obj_rect_bot_left, obj_rect_top_right = (obj_rect[2], obj_rect[1]), (obj_rect[3], obj_rect[0])  # x, y
            obj_rect_w, obj_rect_h = obj_rect_top_right[0] - obj_rect_bot_left[0], obj_rect_bot_left[1] - obj_rect_top_right[1]
            obj_rect_true_center = [(obj_rect_bot_left[0] + int(obj_rect_w/2)), (obj_rect_top_right[1] + int(obj_rect_h/2))] # x, y

            possible_homographies = []
            query_img_copy = query_img.copy()
            for obj_frame_path in obj_frame_paths:
                draw_img = query_img.copy()

                # For each frame of a specific object
                obj_frame = cv2.imread(obj_frame_path, cv2.IMREAD_GRAYSCALE) 
                obj_frame = resize_with_pad(obj_frame, (250, 250))
                
                obj_kp, obj_des = sift.detectAndCompute(obj_frame, None)

                good_matches = match_descriptors(bf, obj_des, query_des, ratio_test=0.75, k=2)

                if len(good_matches) > 10:
                    src_pts = np.float32([ obj_kp[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
                    dst_pts = np.float32([ query_kp[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
                    matchesMask = mask.ravel().tolist()

                    if M is not None:
                        h,w = obj_frame.shape
                        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                        dst = cv2.perspectiveTransform(pts,M)
                        draw_img = cv2.polylines(draw_img,[np.int32(dst)],True,255,3, cv2.LINE_AA)

                        M_err = homography_reprojection_err(M, src_pts, dst_pts)
                        possible_homographies.append((M_err, M, good_matches, matchesMask, obj_frame, obj_kp, obj_des))
                    else:
                        matchesMask = None
                else:
                    matchesMask = None

                draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

                draw_img = cv2.drawMatches(obj_frame, obj_kp, draw_img, query_kp, good_matches, None,**draw_params)

                cv2.imshow('Matches', draw_img)
                cv2.waitKey(10)
            
            # There was an object detection
            if len(possible_homographies) > 0:
                sorted_homographes = sorted(possible_homographies, key=lambda x: (len(x[2]) * -1, x[0]))
                M_best_err, M_best, M_best_matches, M_best_matches_mask, M_best_frame, M_best_frame_kp, M_best_frame_des =  sorted_homographes[0]

                h,w = M_best_frame.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,M_best)

                test = [x[0] for x in dst]
                x, y = [a[0] for a in test], [a[1] for a in test]
                bot_left, top_right = (int(np.min(x)), int(np.max(y))), (int(np.max(x)), int(np.min(y)))
                width, height = abs(top_right[0] - bot_left[0]), abs(bot_left[1] - top_right[1])
                center = (bot_left[0] + int(width/2), top_right[1] + int(height/2))

                dist_from_true = math.dist(center, obj_rect_true_center)

                img_w_bound_box = cv2.rectangle(query_img.copy(), bot_left, top_right, (255, 255, 255), 1, cv2.LINE_AA)
                img_w_bound_box = cv2.circle(img_w_bound_box, center, 5, (255, 255, 255), -1)
                img_w_bound_box = cv2.circle(img_w_bound_box, obj_rect_true_center, 5, (0, 0, 0), -1)

                draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = M_best_matches_mask, # draw only inliers
                    flags = 2)
                show_img = cv2.drawMatches(M_best_frame, M_best_frame_kp, img_w_bound_box, query_kp, M_best_matches, None, **draw_params)
                cv2.imshow('Matches', show_img)
                cv2.waitKey() & 0xFF == ord('q')

                if obj_path in obj_stats:
                    obj_stats[obj_path]["correct"].append(dist_from_true)
                else:
                    obj_stats[obj_path] = {"correct": [dist_from_true], "incorrect": 0}
            else:
                # No object detected
                if obj_path in obj_stats:
                    obj_stats[obj_path]["incorrect"] += 1
                else:
                    obj_stats[obj_path] = {"correct": [], "incorrect": 1}
            print(obj_stats)
    
    output_stats(obj_stats)