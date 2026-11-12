"""
extract_landmarks.py — Step 1 of the BdSL paper pipeline.

Reads word-labelled video clips and turns each one into a .npy file of
MediaPipe Tasks (Pose + Hand) landmarks (225 numbers per frame: 33 pose + 21 left
hand + 21 right hand, each x/y/z; face landmarks are NOT used — same
choice as the official BdSLW60 reference code).

Expected input layout (BdSLW60 videoclips v2 unzips exactly like this):

    data/
        Ami/
            W001_U01_001.mp4
            W001_U02_003.mp4
            ...
        Valo/
            ...
        ...  (one folder per word; folder name = label)

The custom 9-word set uses the same layout — just 9 folders.

Signer IDs are read from the file name when possible. The BdSLW60
convention is U01..U18 (e.g. "..._U04_..."). If a file name has no
signer code, signer is recorded as "unknown" and the training script
falls back to a random split (with a loud warning).

Output (in --out-dir):
    <word>/<clip_id>.npy      one float32 array per clip, shape (T, 225)
    manifest.csv              clip_id, word, signer, n_frames,
                              n_detected_frames, path

Example:
    python extract_landmarks.py --data-dir ./data --out-dir ./features
    python extract_landmarks.py --data-dir ./data --out-dir ./features \\
        --words Ami Valo Achi --force
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np

# Imported from the same folder.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (FEATURE_DIM, WORDS_9, extract_frame_features,
                      fill_gaps, make_holistic, normalize_sequence)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Signer codes look like U04, S12, signer_3 ... (case-insensitive).
# BdSLW60 videoclips v2 files start with the signer code directly:
# U10W37F_trial_0_L.mp4 -> signer U10 (no separator before it, and
# the code is followed by W37, not a separator — so the leading-U
# pattern must be tried FIRST).
SIGNER_PATTERNS = [
    re.compile(r"^(U\d+)", re.IGNORECASE),
    re.compile(r"[_\-.](U\d+)[_\-.]", re.IGNORECASE),
    re.compile(r"[_\-.](S\d+)[_\-.]", re.IGNORECASE),
    re.compile(r"signer[_\-]?(\d+)", re.IGNORECASE),
]


def parse_signer(filename, signer_regex=None):
    """Return the signer code from a file name, or 'unknown'."""
    if signer_regex is not None:
        m = re.search(signer_regex, filename)
        if m:
            return m.group(1)
        return "unknown"
    for pat in SIGNER_PATTERNS:
        m = pat.search(filename)
        if m:
            code = m.group(1).upper()
            return code if code.startswith("U") else "U" + code.lstrip("S")
    return "unknown"


def process_video(video_path, holistic, max_frames=None):
    """Run MediaPipe Tasks landmarkers over one video.

    Returns (seq, n_detected): seq is a float32 array (T, 225),
    already normalised and gap-filled; n_detected is how many frames
    had at least one landmark set.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    frames, detected = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        vec, found = extract_frame_features(results)
        frames.append(vec)
        detected.append(found)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()

    if not frames:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32), 0

    seq = np.stack(frames).astype(np.float32)
    detected = np.array(detected, dtype=bool)
    seq = fill_gaps(seq, detected, max_gap=5)
    seq = normalize_sequence(seq)
    return seq, int(detected.sum())


def main():
    parser = argparse.ArgumentParser(
        description="Extract 225-dim MediaPipe Tasks landmarks from "
                    "word-labelled sign videos.")
    parser.add_argument("--data-dir", required=True,
                        help="Folder with one subfolder per word.")
    parser.add_argument("--out-dir", required=True,
                        help="Where .npy files and manifest.csv go.")
    parser.add_argument("--words", nargs="*", default=None,
                        help="Only process these word folders "
                             "(default: every folder found).")
    parser.add_argument("--signer-regex", default=None,
                        help="Custom regex with one group to read the "
                             "signer code from file names.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Cap frames per clip (for a quick test run).")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if the .npy already exists.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    word_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    if args.words:
        wanted = set(args.words)
        word_dirs = [d for d in word_dirs if d.name in wanted]

    manifest_path = out_dir / "manifest.csv"
    rows = []
    if manifest_path.exists() and not args.force:
        with open(manifest_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    with make_holistic() as holistic:
        done = 0
        for word_dir in word_dirs:
            word = word_dir.name
            videos = sorted(v for v in word_dir.iterdir()
                            if v.suffix.lower() in VIDEO_EXTS)
            for video in videos:
                clip_id = video.stem
                out_file = out_dir / word / f"{clip_id}.npy"
                if out_file.exists() and not args.force:
                    continue
                out_file.parent.mkdir(parents=True, exist_ok=True)
                try:
                    seq, n_detected = process_video(
                        video, holistic, max_frames=args.max_frames)
                except Exception as exc:  # noqa: BLE001 - report and move on
                    print(f"[skip] {video}: {exc}")
                    continue
                if seq.shape[0] == 0:
                    print(f"[skip] {video}: no frames read")
                    continue
                np.save(out_file, seq)
                rows.append({
                    "clip_id": clip_id,
                    "word": word,
                    "signer": parse_signer(video.name, args.signer_regex),
                    "n_frames": seq.shape[0],
                    "n_detected_frames": n_detected,
                    "path": str(out_file),
                })
                done += 1
                if done % 50 == 0:
                    print(f"  ... {done} clips done")

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["clip_id", "word", "signer",
                           "n_frames", "n_detected_frames", "path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone: {done} new clips extracted, {len(rows)} total in manifest.")
    print(f"Manifest: {manifest_path}")
    # Quick per-word summary.
    from collections import Counter
    counts = Counter(r["word"] for r in rows)
    for word in sorted(counts):
        print(f"  {word}: {counts[word]} clips")
    no_detect = sum(1 for r in rows if int(r["n_detected_frames"]) == 0)
    if no_detect:
        print(f"WARNING: {no_detect} clips had ZERO detected frames "
              "(no person found — check the video).")


if __name__ == "__main__":
    main()
