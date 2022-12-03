import cv2

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

def status(msg):
    print(f"\r{msg}: ", end="", flush=True)

# reduce a list of frames to only `k` frames with KMeansClutering
def cluster_frames(frames, k):
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

