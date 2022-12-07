# Simply turn a video into a directory full of still images

import cv2
import os
import argparse

def video_to_frames(video, path_output_dir, prefix):
    # extract frames from a video and save to directory as 'x.png' where 
    # x is the frame index
    vidcap = cv2.VideoCapture(video)
    count = 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            cv2.imwrite(os.path.join(path_output_dir, prefix+'%d.png') % count, image)
            count += 1
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dump video frames to directory')
    parser.add_argument('input', help='Path to video to convert')
    parser.add_argument('output', help='output directory for image files')
    parser.add_argument('--prefix', default='')
    args = parser.parse_args()
    video_to_frames(args.input, args.output, args.prefix)