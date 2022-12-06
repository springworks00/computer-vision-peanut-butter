import cv2
import numpy as np
from typing import NamedTuple
from typing import Iterable

class Frame(NamedTuple):
    raw: np.ndarray = None
    kps: np.ndarray = None
    dcs: np.ndarray = None

def capture(path: str) -> Iterable[Frame]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    cap_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def gen():
        while cap.isOpened():
            ok, raw = cap.read()
            if not ok:
                break
            yield Frame(raw=raw)

    return gen(), cap_len


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

