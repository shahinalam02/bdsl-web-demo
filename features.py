"""
features.py — shared landmark feature code for the BdSL paper pipeline.

What this file does (in plain words):
  * Takes one video frame -> runs MediaPipe Tasks landmarkers -> returns
    225 numbers.
  * 225 = 33 pose landmarks x 3 (x, y, z)  =  99
       + 21 left-hand landmarks x 3         =  63
       + 21 right-hand landmarks x 3        =  63
    Face landmarks (468) are NOT used — the official BdSLW60 reference code
    (github.com/hasanssl/BdSLW60_Code) has the same 225-feature setup
    ("Pose, and Hand Landmarks, NO FACE") and it is much lighter/faster.
  * Normalises each clip: centre on the torso (midpoint of the shoulders)
    and divide by shoulder width, so the same sign looks the same
    whatever the signer's size or position in the frame.
  * Fills short gaps (a few frames where MediaPipe lost the hands)
    by straight-line interpolation, so the model never sees a jump.

Extractor backend (updated 12 Nov 2026 — read this before changing it):
  The old code used `mp.solutions.holistic` (MediaPipe "legacy solutions").
  That API no longer exists: it is gone from every mediapipe version
  that installs on current Colab (verified 12 Nov 2026 — `mp.solutions`
  is missing from both 0.10.30 and 0.10.35), and the old pin
  mediapipe==0.10.14 has no wheel for Colab's Python at all
  ("No matching distribution for mediapipe==0.10.14").
  The extractor now uses the MediaPipe **Tasks API** instead:
  `PoseLandmarker` (33 pose landmarks) + `HandLandmarker` (21 landmarks
  per hand, up to 2 hands). The output is byte-for-byte the same
  225-dim layout, and `make_holistic()` / `extract_frame_features()`
  keep their old interface, so `extract_landmarks.py`, `train.py` and
  `realtime.py` did not need to change.
  Pin: mediapipe==0.10.35 (verified 12 Nov 2026).
  The landmarkers need two model files (downloaded once, see below):
  `pose_landmarker_lite.task` and `hand_landmarker.task`.

The 9 thesis words and their Bangla spellings live here too, so every
script in this folder uses the same labels.
"""

import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_POSE = 33            # MediaPipe pose landmarks
N_HAND = 21            # MediaPipe hand landmarks (per hand)
FEATURE_DIM = (N_POSE + 2 * N_HAND) * 3   # 225

# Pose landmark indices (MediaPipe Pose): 11 = left shoulder, 12 = right
# shoulder, 23 = left hip, 24 = right hip.
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

# Model files for the Tasks-API landmarkers.
POSE_MODEL_NAME = "pose_landmarker_lite.task"
HAND_MODEL_NAME = "hand_landmarker.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task")

# The 9 words of the thesis proposal, in the order used for labels.
WORDS_9 = ["Ami", "Valo", "Achi", "Dhonnobad", "Tumi",
           "Kemon", "Acho", "Vat", "Kheyecho"]

# English label -> Bangla script (used for on-screen text / speech).
WORD_TO_BANGLA = {
    "Ami": "আমি",
    "Valo": "ভালো",
    "Achi": "আছি",
    "Dhonnobad": "ধন্যবাদ",
    "Tumi": "তুমি",
    "Kemon": "কেমন",
    "Acho": "আছো",
    "Vat": "ভাত",
    "Kheyecho": "খেয়েছো",
}

# Known full sentences built from the 9 words. The language-correction
# layer in realtime.py matches these patterns first; anything else is
# joined word-by-word with a full stop.
SENTENCE_TEMPLATES = [
    (("Ami", "Valo", "Achi"), "আমি ভালো আছি।"),
    (("Tumi", "Kemon", "Acho"), "তুমি কেমন আছো?"),
    (("Ami", "Valo", "Achi", "Dhonnobad"), "আমি ভালো আছি, ধন্যবাদ।"),
    (("Tumi", "Vat", "Kheyecho"), "তুমি ভাত খেয়েছো?"),
    (("Ami", "Vat", "Kheyecho"), "আমি ভাত খেয়েছি।"),
    (("Dhonnobad",), "ধন্যবাদ।"),
]


# ---------------------------------------------------------------------------
# Tasks-API extractor (replaces mp.solutions.holistic)
# ---------------------------------------------------------------------------
class _Point:
    """One landmark, mimicking MediaPipe's NormalizedLandmark (.x/.y/.z)."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)


class _LandmarkSet:
    """Mimics a MediaPipe landmark list object: `.landmark` -> points."""

    def __init__(self, points):
        self.landmark = points


class _Results:
    """Mimics mp.solutions.holistic results: pose/left/right or None."""

    def __init__(self, pose, left, right):
        self.pose_landmarks = pose
        self.left_hand_landmarks = left
        self.right_hand_landmarks = right


def _find_model(explicit, name, url):
    """Locate a .task model file, or explain how to get it.

    Search order: explicit path -> this file's folder -> working
    directory -> ./models/. Raises FileNotFoundError with the download
    URL if it is nowhere to be found.
    """
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    here = Path(__file__).resolve().parent
    candidates.extend([here / name, Path.cwd() / name,
                       Path.cwd() / "models" / name])
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"Model file '{name}' not found. Download it once with:\n"
        f"  wget -q {url} -O {name}\n"
        f"(or place it next to features.py).")


class TasksHolistic:
    """Pose + hand landmark extraction via the MediaPipe Tasks API.

    Same interface as the old `mp.solutions.holistic.Holistic` object
    used to have: `.process(rgb_frame)` returns a results object with
    `.pose_landmarks`, `.left_hand_landmarks` and
    `.right_hand_landmarks` (each a landmark list or None), and the
    object works as a context manager. `extract_frame_features()`
    below consumes it unchanged.
    """

    # VIDEO running mode needs strictly increasing timestamps (ms).
    _FRAME_MS = 33  # ~30 fps

    def __init__(self, pose_model_path=None, hand_model_path=None,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        pose_path = _find_model(pose_model_path, POSE_MODEL_NAME,
                                POSE_MODEL_URL)
        hand_path = _find_model(hand_model_path, HAND_MODEL_NAME,
                                HAND_MODEL_URL)
        pose_options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(pose_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=min_tracking_confidence)
        hand_options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(hand_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=min_tracking_confidence)
        self._mp = mp
        self._pose = vision.PoseLandmarker.create_from_options(pose_options)
        self._hand = vision.HandLandmarker.create_from_options(hand_options)
        self._timestamp_ms = 0

    def process(self, rgb):
        """Run both landmarkers on one RGB frame -> results object."""
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts = self._timestamp_ms
        self._timestamp_ms += self._FRAME_MS
        pose_res = self._pose.detect_for_video(mp_image, ts)
        hand_res = self._hand.detect_for_video(mp_image, ts)

        pose = None
        if pose_res.pose_landmarks:
            pose = _LandmarkSet(
                [_Point(lm.x, lm.y, lm.z)
                 for lm in pose_res.pose_landmarks[0]])

        left = right = None
        for hand_lms, handed in zip(hand_res.hand_landmarks,
                                    hand_res.handedness):
            label = handed[0].category_name if handed else ""
            pts = _LandmarkSet(
                [_Point(lm.x, lm.y, lm.z) for lm in hand_lms])
            if label == "Left":
                left = pts
            elif label == "Right":
                right = pts
            elif left is None:
                left = pts
            elif right is None:
                right = pts
        return _Results(pose, left, right)

    def close(self):
        self._pose.close()
        self._hand.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def make_holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                  pose_model_path=None, hand_model_path=None):
    """Create the Tasks-API landmark extractor (see TasksHolistic).

    Model files are found next to features.py, in the working directory,
    or in ./models/ — download them once (URLs are in the
    FileNotFoundError message if they are missing).
    """
    return TasksHolistic(
        pose_model_path=pose_model_path,
        hand_model_path=hand_model_path,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence)


# ---------------------------------------------------------------------------
# Per-frame extraction
# ---------------------------------------------------------------------------
def extract_frame_features(results):
    """Turn one extractor result into a 225-dim vector.

    Args:
        results: output of TasksHolistic.process(rgb_frame).

    Returns:
        (vec, detected): vec is a float32 array of shape (225,);
        detected is True if at least one landmark set was found.
        Missing parts are zeros — the caller can interpolate them later.
    """
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
    detected = False

    if results.pose_landmarks is not None:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            vec[i * 3 + 0] = lm.x
            vec[i * 3 + 1] = lm.y
            vec[i * 3 + 2] = lm.z
        detected = True

    offset = N_POSE * 3  # 99
    if results.left_hand_landmarks is not None:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            vec[offset + i * 3 + 0] = lm.x
            vec[offset + i * 3 + 1] = lm.y
            vec[offset + i * 3 + 2] = lm.z
        detected = True

    offset += N_HAND * 3  # 162
    if results.right_hand_landmarks is not None:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            vec[offset + i * 3 + 0] = lm.x
            vec[offset + i * 3 + 1] = lm.y
            vec[offset + i * 3 + 2] = lm.z
        detected = True

    return vec, detected


# ---------------------------------------------------------------------------
# Sequence-level processing
# ---------------------------------------------------------------------------
def normalize_sequence(seq):
    """Centre on the torso and scale by shoulder width, frame by frame.

    Args:
        seq: float32 array of shape (T, 225).

    Returns:
        Normalised copy of the same shape. Frames with no detection
        (all zeros) are left as zeros.
    """
    seq = np.array(seq, dtype=np.float32, copy=True)
    if seq.shape[0] == 0:
        return seq

    pose = seq[:, :N_POSE * 3].reshape(-1, N_POSE, 3)      # (T, 33, 3)
    left_sh = pose[:, LEFT_SHOULDER, :2]                    # (T, 2)
    right_sh = pose[:, RIGHT_SHOULDER, :2]                  # (T, 2)

    centre = (left_sh + right_sh) / 2.0                     # (T, 2)
    scale = np.linalg.norm(left_sh - right_sh, axis=1)      # (T,)

    # Frames where the shoulders were not found get scale 0; use the
    # clip's median shoulder width instead (or 1.0 if there is none).
    good = scale > 1e-6
    fallback = np.median(scale[good]) if good.any() else 1.0
    scale = np.where(good, scale, fallback)
    centre = np.where(good[:, None], centre, 0.0)

    # Apply to x, y of every landmark; z is divided by the same scale
    # (it is a relative depth, so it only needs the size correction).
    seq[:, 0::3] -= np.repeat(centre[:, 0:1], FEATURE_DIM // 3, axis=1)
    seq[:, 1::3] -= np.repeat(centre[:, 1:2], FEATURE_DIM // 3, axis=1)
    seq /= scale[:, None]
    return seq


def fill_gaps(seq, detected, max_gap=5):
    """Linearly interpolate short runs of undetected frames.

    Args:
        seq: float32 array of shape (T, 225).
        detected: boolean array of shape (T,) — True where MediaPipe
            found at least one landmark set.
        max_gap: only fill gaps of this many frames or fewer.

    Returns:
        A copy of seq with short gaps filled. Long gaps (e.g. the
        signer left the frame) are left as zeros on purpose.
    """
    seq = np.array(seq, dtype=np.float32, copy=True)
    T = seq.shape[0]
    i = 0
    while i < T:
        if detected[i]:
            i += 1
            continue
        j = i
        while j < T and not detected[j]:
            j += 1
        gap = j - i  # undetected frames i .. j-1
        if 0 < gap <= max_gap and i > 0 and j < T:
            # Straight line between the last good frame and next good one.
            for k in range(i, j):
                alpha = (k - i + 1) / (gap + 1)
                seq[k] = (1.0 - alpha) * seq[i - 1] + alpha * seq[j]
        i = j
    return seq


def hand_motion(prev_vec, curr_vec):
    """How much the hands moved between two frames.

    Only the hand part of the vector (dims 99..225) is used, because
    body sway should not count as signing. Used for pause detection
    in realtime.py.
    """
    if prev_vec is None or curr_vec is None:
        return 0.0
    return float(np.mean(np.abs(curr_vec[N_POSE * 3:] - prev_vec[N_POSE * 3:])))
