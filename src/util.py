import cv2
import numpy as np
from typing import NamedTuple
from typing import Iterable

class Frame(NamedTuple):
    raw: np.ndarray = None
    kps: np.ndarray = None
    dcs: np.ndarray = None

def status(msg):
    print(f"\r{msg}", end="", flush=True)

def capture(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    cap_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    i = 1
    while cap.isOpened():
        ok, raw = cap.read()
        if not ok:
            break
        frames.append(Frame(raw=raw))
        status(f"\r({path}) reading frames: {100*i//cap_len}%")
        i += 1
    print()
    return frames

def error_frame(shape):
    black = np.zeros(shape=shape, dtype=np.uint8)
    black = cv2.putText(
            black, 
            "OBJECT NOT FOUND",
            (100, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            3,
            cv2.LINE_AA)
    return Frame(raw=black)

def show(frame: Frame, kps=False, scale=1, title="untitled"):
    raw = frame.raw
    if kps:
        raw = cv2.drawKeypoints(
            raw, frame.kps, 0, color=(0, 0, 255),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    width = int(raw.shape[1] * scale)
    height = int(raw.shape[0] * scale)
    raw = cv2.resize(raw, (width, height))

    cv2.imshow(title, raw)

def destroy(title="untitled"):
    cv2.destroyWindow(title)

def key_pressed(key, wait=False):
    if wait:
        return cv2.waitKey() & 0xFF == ord(key)
    else:
        return cv2.waitKey(1) & 0xFF == ord(key)

