"""Step 5: render and cache every recording at one location, then write the split and normalisation stats.

    python scripts/build_cache.py --data_dir /path/to/rounD/data --out_dir $SCRATCH/round_cache
    python scripts/build_cache.py ... --recs 17 18      # only some recordings (quick test)

Output:
    out_dir/rec_NN/{frames.npy, states.npy, actions.npy, windows.npy, tracks.csv, info.json}
    out_dir/split.json   train / val / test recording ids
    out_dir/stats.json   speed and action mean/std from the TRAIN recordings
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roundabout_wm.build import build_recording, split_recordings
from roundabout_wm.config import Config
from roundabout_wm.dataset import compute_stats, save_stats
from roundabout_wm.round_io import list_recordings, load_recording


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--recs", type=int, nargs="*", default=None)
    p.add_argument("--skip_existing", action="store_true")
    a = p.parse_args()
    cfg = Config()

    recs = a.recs or list_recordings(a.data_dir, cfg.data.location_id)
    print(f"recordings at location {cfg.data.location_id}: {recs}")
    for r in recs:
        out = os.path.join(a.out_dir, f"rec_{r:02d}")
        if a.skip_existing and os.path.exists(os.path.join(out, "info.json")):
            print(f"rec {r}: exists, skipping"); continue
        t0 = time.time()
        rec = load_recording(a.data_dir, r, step=cfg.data.step)
        info = build_recording(rec, cfg, out)
        print(f"rec {r}: {info} ({time.time() - t0:.0f} s)")

    if a.recs is None:
        split = split_recordings(recs, cfg)
        with open(os.path.join(a.out_dir, "split.json"), "w") as f:
            json.dump(split, f, indent=1)
        save_stats(compute_stats(a.out_dir, split["train"], cfg), os.path.join(a.out_dir, "stats.json"))
        print("split:", split)


if __name__ == "__main__":
    main()
