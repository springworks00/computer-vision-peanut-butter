import cv2
import numpy as np
import sklearn

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from typing import Iterable
from typing import NamedTuple
from typing import List
import matplotlib.pyplot as plt

class Frame(NamedTuple):
    raw: np.ndarray = None
    kps: np.ndarray = None
    dcs: np.ndarray = None

def status(msg):
    print(f"\r{msg}: ", end="", flush=True)

# reduce a list of frames to only `k` frames with KMeansClutering
def cluster_frames_v1(frames, k):
    # compute the descriptors for all the frames
    frames_dcss = []
    for i, f in enumerate(frames):
        _, dcs = ORB.detectAndCompute(f, None)
        frames_dcss.append(dcs)
        status(f"analyzing {i+1}/{len(frames)} frames for reduction")
    print("done")

    print(f"clustering {len(frames)} frames to {k} frames")
    unique_dcs = np.unique([x for xs in frames_dcss for x in xs])

    points = []
    for u_dc in unique_dcs:
        for frame_i, frame_dcs in enumerate(frames_dcss):
            if u_dc in frame_dcs:
                points.append([u_dc, frame_i])

    # K-Means Cluster the unique descriptor occurences
    np_points = np.array(points)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(np_points)

    best_indexes = pairwise_distances_argmin_min(kmeans.cluster_centers_, np_points)[0]
    best_indexes = [points[bi][1] for bi in best_indexes]

    new_fs = [frames[bi] for bi in best_indexes]
    new_fs_dcss = [frames_dcss[bi] for bi in best_indexes]
    print("done")

    return new_fs, new_fs_dcss # new frames, and their descriptors



def __match(query: Frame, train: Frame, threshold):
    matches = cv2.BFMatcher().knnMatch(query.dcs, train.dcs, k=2)

    return [m for m,n in matches if m.distance < threshold*n.distance]

def most_similar(query: Frame, trains: Iterable[Frame], confidence_floor, threshold) -> Frame:
    nfeats=len(query.dcs)

    best_score = -1
    best = None
    for t in trains:
        this_score = len(__match(query, t, threshold=threshold))
        if this_score < nfeats*confidence_floor:
            # not enough matches to assert the object is even present
            continue
        if this_score > best_score:
            best_score = this_score
            best = t
    #print(best_score, f"(required = {nfeats*confidence_floor})")
    return best

def cluster_frames_v2(frames: List[Frame], k) -> List[Frame]:
    dcss = list(map(lambda x: x.dcs, frames))

    xs, ys = [], []
    train = np.array(dcss[0])

    len_dcss = len(dcss)
    for frameIdx, dcs in enumerate(dcss):
        this_ys, train = __acc_match(dcs, train)
        this_xs = [frameIdx]*len(this_ys)
        
        ys += this_ys
        xs += this_xs
    points = np.array(list(zip(xs, ys)))
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(points)
    best_indexes = pairwise_distances_argmin_min(kmeans.cluster_centers_, points)[0]
    return list(map(lambda i: frames[i], points[best_indexes, 0]))