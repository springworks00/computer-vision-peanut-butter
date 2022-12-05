import numpy as np
import cv2 as cv
import argparse
import os

def test_point(bb, pt) -> bool:
    return pt[0] >= bb[0] and pt[1] >= bb[1]\
        and pt[0] <= bb[0] + bb[2] and pt[1] <= bb[1] + bb[3]

def clip_bb(bb, size) -> np.array:
    bb = bb.copy()
    x,y,w,h = bb
    if x < 0:
        w += x
        x = 0
    if y < 0:
        h += y
        y = 0
    if x + w > size[0]:
        over = size[0] - (x + w)
        w += over
    if y + h > size[1]:
        over = size[1] - (y + h)
        h += over
    return np.array([x,y,w,h])

def save_annotations_file(filename : str, annotations, store_mode : bool) -> None:
    with open(filename, 'a' if store_mode else 'w') as f:
        for file, bb in annotations:
            line = '%s %i %i %i %i %i \n' % (file, 1, *bb)
            f.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
                                                The example file can be downloaded from: \
                                                https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
    parser.add_argument('image', type=str, help='path to image file')
    parser.add_argument('--centroid_f',type=float, help='blending weight for how quickly the bounding box should adjust to new centroid of features', default=0.5)
    parser.add_argument('--tgtbb_f', type=float, default=0.1)
    parser.add_argument('--prefix', default='')
    parser.add_argument('--output_images', help='optional directory to output image files from video.\
                                                if this argument is supplied, the frames in the video will be saved.')
    parser.add_argument('--output_annotations', help='annotations file for training. If supplied, a file with all \
                                                    bounding box (x,y,w,h) coordinates and their corresponding files.\
                                                    --output_images must be provided if this argument is used.')
    parser.add_argument('--append', action='store_true')
    args = parser.parse_args()

    if not args.output_annotations is None:
        if args.output_images is None:
            print('Error: --output_images must be used with --output_annotations')
            exit(-1)
        if not os.path.isdir(args.output_images):
            print('Error: --output_images must be a directory')
            exit(-1)

        annotations_dirname = os.path.dirname(args.output_annotations)
        annotations_relpath = os.path.relpath(args.output_images, annotations_dirname)
        print('Annotations relative file path is %s' % annotations_relpath)


    cv.namedWindow('frame', flags=cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    cv.resizeWindow('frame', 500, 500)

    matcher = cv.BFMatcher_create()

    N_TRACKED_CORNERS = 50

    cap = cv.VideoCapture(args.image)
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = N_TRACKED_CORNERS,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
                    minEigThreshold=1e-3
                    )
    # Create some random colors
    color = np.random.randint(0, 255, (N_TRACKED_CORNERS, 3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    bb = np.array(cv.boundingRect(p0))

    annotations = []

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    frame_id = 0
    while(1):
        ret, frame = cap.read()
        if not ret:
            print('No frames grabbed!')
            break
        frame_gray = cv.medianBlur(frame, 5)
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


        # calculate centroid
        centroid = np.mean(p0, axis=0)[0]

        std = np.std(p0, axis=0)[0]

        if len(p0) < N_TRACKED_CORNERS:
            print("Finding new points")
            points_proposed = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
            matches = matcher.match(np.reshape(points_proposed, (-1,2)).astype(np.float32), np.reshape(p0, (-1,2)).astype(np.float32))

            # find new points by checking their distance from existing ones. If they are far enough, add them
            good = []
            for m in matches:
                if m.distance > 10:
                    point = points_proposed[m.queryIdx]
                    tpoint = point.ravel()
                    if test_point(bb, tpoint) or np.linalg.norm(point - centroid) < 2 * np.linalg.norm(std):
                        good.append(point)
            if len(good) > 0:
                good = np.stack(good, axis=0)
                p0 = np.concatenate([p0, good], axis=0)
        
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]

        vel = good_new - good_old
        avg_vel = np.mean(vel, axis=0)

        # adjust center of bounding box towards feature point cluster
        bb[:2] = bb[:2] + avg_vel

        c_xy = bb[:2] + (bb[2:] // 2)
        dc_xy = centroid - c_xy

        bb[:2] = bb[:2] + args.centroid_f * dc_xy

        # adjust bounding box size towards box enclosing all good points
        bb_target = np.array(cv.boundingRect(good_new))

        bb[2:] = (1-args.tgtbb_f) * bb[2:] + args.tgtbb_f * (1.1 * bb_target[2:])

        print('std: %s centroid: %s avg_vel: %s' % (std, centroid, avg_vel))

        # save frame before we draw all over it
        img_ann_path = ''
        if not args.output_images is None:
            img_filename = args.prefix + '%i.png' % frame_id
            img_path = os.path.join(args.output_images, img_filename)
            img_ann_path = os.path.join(annotations_relpath, img_filename)
            print('Saving frame %s (%s)' % (img_path, img_ann_path))
            cv.imwrite(img_path, frame)
        h,w = frame.shape[:2]
        annotations.append((img_ann_path, clip_bb(bb, (w,h))))

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            # mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i%len(color)].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i%len(color)].tolist(), -1)
        
        rect = annotations[-1][1]
        frame = cv.circle(frame, (int(c_xy[0]), int(c_xy[1])), 15, (0, 255, 0), -1)
        frame = cv.rectangle(frame, np.int32((rect[0], rect[1])), np.int32((rect[0] + rect[2], rect[1] + rect[3])), (255, 0, 0), thickness=2)
        img = cv.add(frame, mask)
        cv.imshow('frame', img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        if k == ord('s'):
            print('Saving frame')
            cv.imwrite('tracker_output.png', frame)
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        frame_id = frame_id + 1
    cv.destroyAllWindows()

    if not args.output_annotations is None:
        save_annotations_file(args.output_annotations, annotations=annotations, store_mode=args.append)