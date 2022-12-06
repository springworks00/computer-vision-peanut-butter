from __future__ import print_function
import cv2 as cv
import argparse

def detectAndDisplay(frame, cascade_classifier : cv.CascadeClassifier):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    ROIs = cascade_classifier.detectMultiScale(frame_gray)
    for (x,y,w,h) in ROIs:
        frame = cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 4)
        # faceROI = frame_gray[y:y+h,x:x+w]
        # #-- In each face, detect eyes
        # eyes = eyes_cascade.detectMultiScale(faceROI)
        # for (x2,y2,w2,h2) in eyes:
        #     eye_center = (x + x2 + w2//2, y + y2 + h2//2)
        #     radius = int(round((w2 + h2)*0.25))
        #     frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
    cv.imshow('Capture - Detection', frame)

def load_cascadeCLassifier(path : str) -> cv.CascadeClassifier:
    cascade_classifier = cv.CascadeClassifier()

    if not cascade_classifier.load(cv.samples.findFile(path)):
        print('--(!)Error loading cascade file %s' % path)
        return None
    
    return cascade_classifier


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cascade Classifier.')
    parser.add_argument('--cascade_file', help='Path to face cascade.', default='venv/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
    parser.add_argument('--camera', help='Camera divide number.', type=str, default=0)
    args = parser.parse_args()
    face_cascade_name = args.cascade_file
    cascade_classifier = load_cascadeCLassifier(face_cascade_name)

    cv.namedWindow('Capture - Detection', flags=cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    cv.resizeWindow('Capture - Detection', 500, 500)
    
    # support camera ind or image sequence
    camera_device = args.camera
    # if camera_device.isnumeric():
    #     camera_device = int(camera_device)

    detectAndDisplay(cv.imread(camera_device), cascade_classifier)
    cv.waitKey(0)

    exit(0)
    
    #-- 2. Read the input
    cap = cv.VideoCapture(camera_device)
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
        detectAndDisplay(frame, cascade_classifier)
        if cv.waitKey(1000) == 27:
            break