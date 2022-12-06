import analyze
import util
import numpy as np
import cv2

def load_scan(path):
    print("__ LOADING SCAN __")
    scan = util.capture(path)

    scan = analyze.orb_features(scan, n=100)

    scan = analyze.cluster(scan, k=17)

    return analyze.orb_features(scan, n=200)


def load_env(path):
    print("__ LOADING ENV __")
    env = util.capture(path)
    
    return analyze.orb_features(env, n=100)

scan = load_scan("scan.mp4")
env = load_env("strafe_left.qt")

for i, e in enumerate(env):
    if i % 30 == 0:
        similar = analyze.most_similar(
                e, scan, confidence_floor=0.1, threshold=0.75)
        if similar is None:
            similar = util.error_frame(e.raw.shape)

    util.show(e, scale=0.3, title="env", kps=True)
    util.show(similar, scale=0.3, title="guess", kps=True)

    if util.key_pressed("q", wait=False):
        break

