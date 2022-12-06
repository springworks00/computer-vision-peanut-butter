import cv2 as cv
import argparse

    # https://www.cs.ubc.ca/~lowe/papers/brown05.pdf
        # 3D reconstruction

    # Should we use SIFT?
        # ORB is faster but SIFT could be more accurate for images where the object in question isn't in the main view

    # 1. Narrow down search space based upon where the keypoints are most frequently detected?
        # a. Create kernels over these spaces and rerun

    # Grayscale?

    # Tasks
        # 1. Feature extraction from the video
            # Frame by frame feature consistency?
            # Lower n features?

        # 2. Feature extraction from the photo
            # Higher nfeatures 

        # 3. Feature Matching
            # KNN, ratio testing, RANSAC

        # 4. Bounding box around object

    # Optional for update
        # SIFT vs ORB
        # YOLO to reduce search space
        # Clustering for background noise removal
        # YOLO for background noise removal
        # General object detection
            # YOLO training

class Detector():

    def __init__(self, video_path, img_path):
        self.orb = cv.ORB_create()
        self.bf = cv.BFMatcher()
        self.video_path = video_path
        self.img_path = img_path

    def find_matches(self, img_desc, vid_desc):
        matches = self.bf.knnMatch(img_desc, vid_desc, k=2)

        good = [] # ratio test
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        return good

    def process_video(self, img_img, img_kp, img_desc, match):
        cap = cv.VideoCapture(self.video_path)

        cv.namedWindow('tracked', flags=cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow('tracked', 500, 500)

        while cap.isOpened():
            ret, vid_frame = cap.read()
            if not ret:
                break

            vid_kp, vid_desc = self.orb.detectAndCompute(vid_frame, None)
            matches = self.find_matches(img_desc, vid_desc)

            if match:
                show_img = cv.drawMatchesKnn(img_img, img_kp, vid_frame, vid_kp, matches[:20], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            else:
                show_img = cv.drawKeypoints(vid_frame, vid_kp, None, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            cv.imshow('tracked', show_img)
            if cv.waitKey(100) == 27: # escape
                break

    def process_img(self):
        img_img = cv.imread(self.img_path)

        cv.namedWindow('find', flags=cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow('find', 500, 500)

        img_kp, img_desc = self.orb.detectAndCompute(img_img, None)

        show_img = cv.drawKeypoints(img_img, img_kp, None, color = (0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        cv.imshow('find', show_img)
        cv.waitKey()

        return img_img, img_kp, img_desc

    def run(self):
        img_img, img_kp, img_desc = self.process_img()
        self.process_video(img_img, img_kp, img_desc, match=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', required=True)
    parser.add_argument('--detect_image', required=True)
    parser.add_argument('--tracker_output', default="output.mov")
    args = parser.parse_args()

    det = Detector(args.input_video, args.detect_image)
    det.run()