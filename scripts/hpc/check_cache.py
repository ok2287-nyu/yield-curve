"""Summarise a built cache: recordings, split, windows per split for several rollout lengths, disk use.

    python scripts/hpc/check_cache.py /scratch/$USER/yield/round_cache
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from roundabout_wm.config import Config
from roundabout_wm.dataset import RounDWindows


def main(cache):
    cfg = Config()
    recs = sorted(int(os.path.basename(d)[4:]) for d in glob.glob(os.path.join(cache, "rec_*"))
                  if os.path.exists(os.path.join(d, "info.json")))
    infos = [json.load(open(os.path.join(cache, f"rec_{r:02d}", "info.json"))) for r in recs]
    gb = sum(os.path.getsize(f) for f in glob.glob(os.path.join(cache, "rec_*", "*"))) / 1e9
    print(f"recordings: {len(recs)} {recs}")
    print(f"ego tracks: {sum(i['n_ego_tracks'] for i in infos)}, frames: {sum(i['n_rows'] for i in infos)}, "
          f"disk: {gb:.1f} GB")

    split_path = os.path.join(cache, "split.json")
    split = json.load(open(split_path)) if os.path.exists(split_path) else {"all": recs}
    print("split:", split)
    for n_pred in (15, 20, 30, 50):
        counts = {k: len(RounDWindows(cache, ids, cfg, n_pred=n_pred)) for k, ids in split.items()}
        print(f"  n_pred={n_pred:2d} ({n_pred * cfg.data.dt:4.1f} s): {counts}")

    item = RounDWindows(cache, split[next(iter(split))], cfg, n_pred=20)[0]
    print("one item:", {k: tuple(v.shape) for k, v in item.items()})
    if os.path.exists(os.path.join(cache, "stats.json")):
        print("stats:", json.load(open(os.path.join(cache, "stats.json"))))


if __name__ == "__main__":
    main(sys.argv[1])
