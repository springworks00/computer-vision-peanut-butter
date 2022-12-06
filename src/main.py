import analyze
import util

def scan_pipeline():
    scan, _ = util.capture("scan.mp4")

    print("cluster scan to k orientations")
    scan = analyze.orb_features(scan, n=20)
    scan = analyze.cluster(list(scan), k=20)

    print("re-analyze scan with more features")
    scan = analyze.orb_features(scan, n=200)

    return scan

env_path = "strafe_left.qt"
#env_path = "env.mp4"
env = analyze.orb_features(util.capture(env_path)[0], n=100)

scan = list(scan_pipeline())

similar = scan[0]
fail = 0
for i, e in enumerate(env):
    if i % 30 == 0:
        tmp = analyze.most_similar(
                e, scan, confidence_floor=0.1, threshold=0.75)
        if tmp is None:
            fail += 1
            print(f"\robject not present count: {fail}", end="", flush=True)
        else:
            similar = tmp

    util.show(e, scale=0.3, title="env", kps=True)
    util.show(similar, scale=0.3, title="guess", kps=True)

    if util.key_pressed("q", wait=False):
        break
