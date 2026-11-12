"""
streamlit_app.py — BdSL 10-class (or 60-class) web demo.

Upload an .mp4 sign clip -> MediaPipe Tasks (Pose + Hand) landmarks (225-dim/frame,
same pipeline as features.py / extract_landmarks.py) -> trained PyTorch
checkpoint from train.py -> predicted word + Bangla + confidence + top-3.

Run (after training, see 10-class-quickstart.md Day 5):
    pip install streamlit
    streamlit run streamlit_app.py -- --checkpoint runs/run10/bilstm_best.pt

    # 60-word model works too — the checkpoint carries its own word list:
    streamlit run streamlit_app.py -- --checkpoint runs/run1/bilstm_best.pt

Requirements: same as requirements.txt (torch, mediapipe==0.10.35,
opencv-python-headless, numpy) + streamlit. Run on Colab or a laptop —
this VM has no GPU and no torch/mediapipe installed, so train + demo
run in Colab, not here.

Design note: torch / cv2 / mediapipe / streamlit are imported INSIDE
functions, so the pure-numpy helpers (softmax, top-3, padding) can be
unit-tested without those packages (see test_streamlit_smoke.py).
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from features import FEATURE_DIM  # noqa: E402  (numpy only at import time)

# The 10 quickstart words, in quickstart order.
WORDS_10 = ["Baba", "Maa", "Bhai", "Bon", "Aam",
            "Cha", "Cake", "Doctor", "Juta", "TV"]

# English label -> Bangla script. Covers all 60 BdSLW60 words so the app
# also works with a 60-word checkpoint (words come from the checkpoint).
WORD_TO_BANGLA = {
    "Baba": "বাবা", "Relative": "আত্মীয়", "Bhai": "ভাই", "Bon": "বোন",
    "Wife": "বউ", "Uncle": "চাচা", "Aunt": "চাচী",
    "Grandfather": "দাদা/নানা", "Grandmother": "দাদী/নানী",
    "Responsibility": "দায়িত্ব", "BrotherInLaw": "দেবর",
    "SisterHusband": "দুলাভাই", "Daughter": "কন্যা", "Ma": "মা",
    "Maa": "মা",  # dataset folder is "maa" (W20)
    "Aam": "আম", "Alu": "আলু", "Pineapple": "আনারস", "Grapes": "আঙ্গুর",
    "Apple": "আপেল", "Biscuits": "বিস্কুট", "Jujube": "বরই",
    "Cake": "কেক", "Cha": "চা", "Rice": "চাল", "Sugar": "চিনি",
    "Chips": "চিপস", "Chocolate": "চকলেট", "Lentils": "ডাল",
    "Button": "বোতাম", "Cap": "টুপি", "Shawl": "চাদর", "Comb": "চিরুনি",
    "Spectacles": "চশমা", "Bangles": "চুড়ি", "Clip": "ক্লিপ",
    "Cream": "ক্রিম", "Data": "তথ্য", "Indebted": "দেনাদার",
    "Twin": "জমজ", "Juta": "জুতা", "Toothpaste": "টুথপেস্ট",
    "Tshirt": "টিশার্ট", "Tubelight": "টিউবলাইট", "TV": "টিভি",
    "AC": "এসি", "Apartment": "অ্যাপার্টমেন্ট",
    "AudioCassette": "অডিও ক্যাসেট", "Mirror": "আয়না",
    "Bucket": "বালতি", "Sand": "বালু",
    "AIDS": "এইডস্", "Arthritis": "বাত", "Bandage": "ব্যান্ডেজ",
    "Capsule": "ক্যাপসুল", "Treatment": "চিকিৎসা",
    "Conjunctivitis": "চোখ ওঠা", "Dengue": "ডেঙ্গু",
    "Doctor": "ডক্টর", "Bite": "দংশন", "Weak": "দুর্বল",
}


# ---------------------------------------------------------------------------
# Pure-numpy helpers (testable without torch / mediapipe / streamlit)
# ---------------------------------------------------------------------------
def softmax_np(logits):
    """Softmax over the last axis, numerically stable."""
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def pad_sequence(seq, max_len):
    """Pad (or truncate) a (T, FEATURE_DIM) sequence to (max_len, FEATURE_DIM).

    Returns (padded, length) where length is the real frame count used.
    """
    seq = np.asarray(seq, dtype=np.float32)
    length = min(seq.shape[0], max_len)
    padded = np.zeros((max_len, FEATURE_DIM), dtype=np.float32)
    if length > 0:
        padded[:length] = seq[:length]
    return padded, length


def top3_from_logits(logits, words):
    """Return [(word, prob), ...] for the 3 highest-probability classes."""
    probs = softmax_np(np.asarray(logits, dtype=np.float64).reshape(-1))
    order = np.argsort(-probs)[:3]
    return [(words[i], float(probs[i])) for i in order]


# Lowercase lookup: dataset folder names are lowercase (baba, aam, ...)
# while the map above uses capitalised labels, so look up by lower().
_BANGLA_LOWER = {k.lower(): v for k, v in WORD_TO_BANGLA.items()}


def bangla_label(word):
    """Bangla script for a word, falling back to the word itself."""
    return _BANGLA_LOWER.get(word.lower(), word)


# ---------------------------------------------------------------------------
# Model / video (needs torch, cv2, mediapipe — imported lazily)
# ---------------------------------------------------------------------------
def load_checkpoint(path):
    """Load a train.py checkpoint; returns (model, words, max_len)."""
    import torch
    from models import build_model

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # very old torch without the weights_only argument
        ckpt = torch.load(path, map_location="cpu")
    except Exception:  # checkpoints with non-tensor metadata
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    model = build_model(
        ckpt["model"],
        num_classes=len(ckpt["words"]),
        input_dim=ckpt.get("input_dim", FEATURE_DIM),
        hidden_dim=ckpt.get("hidden_dim", 128),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, list(ckpt["words"]), int(ckpt.get("max_len", 164))


def extract_sequence(video_path):
    """Video file -> (seq, n_detected) via the shared extraction code.

    Same steps as extract_landmarks.process_video: Tasks-API landmarks,
    short-gap interpolation, torso-centred normalisation.
    """
    from extract_landmarks import process_video
    from features import make_holistic

    with make_holistic() as holistic:
        return process_video(str(video_path), holistic)


def predict_sequence(seq, checkpoint_path):
    """Full prediction for one landmark sequence.

    Returns (top3, n_frames) where top3 is [(word, prob), ...].
    Raises ValueError if the clip has no frames at all.
    """
    import torch

    if seq.shape[0] == 0:
        raise ValueError("No frames read from the video.")
    model, words, max_len = load_checkpoint(checkpoint_path)
    padded, length = pad_sequence(seq, max_len)
    x = torch.from_numpy(padded).unsqueeze(0)          # (1, max_len, 225)
    lengths = torch.tensor([length], dtype=torch.long)
    with torch.no_grad():
        logits = model(x, lengths).cpu().numpy()[0]
    return top3_from_logits(logits, words), int(seq.shape[0])


def predict_video(video_path, checkpoint_path):
    """Video file -> (top3, n_frames, n_detected)."""
    seq, n_detected = extract_sequence(video_path)
    top3, n_frames = predict_sequence(seq, checkpoint_path)
    return top3, n_frames, int(n_detected)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    import streamlit as st

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/run10/transformer_best.pt")
    args, _ = parser.parse_known_args()

    st.set_page_config(page_title="BdSL Recognition Demo", layout="centered")
    st.title("Bangla Sign Language Recognition")
    st.write("Upload a short video of one sign — the model predicts the word.")
    st.caption(
        "Research prototype — it can make mistakes. "
        "10-class quickstart model; works with a 60-class checkpoint too."
    )

    ckpt_path = st.text_input("Model checkpoint (.pt)", args.checkpoint)
    uploaded = st.file_uploader("Sign video (.mp4 / .mov / .avi)",
                                type=["mp4", "mov", "avi", "mkv", "webm"])
    if uploaded is None:
        st.info("Upload a video to get a prediction. "
                "Tip: use one of the BdSLW60 clips, or record yourself "
                "signing one of: " + ", ".join(WORDS_10) + ".")
        return

    suffix = Path(uploaded.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name

    with st.spinner("Extracting landmarks and predicting…"):
        try:
            top3, n_frames, n_detected = predict_video(tmp_path, ckpt_path)
        except FileNotFoundError:
            st.error(f"Checkpoint not found: {ckpt_path}. Train first "
                     "(quickstart Day 3–4), then come back.")
            return
        except Exception as exc:  # noqa: BLE001 - show the error, don't crash
            st.error(f"Could not process this video: {exc}")
            return

    if n_detected == 0:
        st.warning("No hands/body detected in this video — check the "
                   "lighting and keep the signer centred in the frame.")
        return

    word, prob = top3[0]
    st.success(f"**Predicted: {word} ({bangla_label(word)})** — "
               f"confidence {prob:.1%}")
    st.write(f"Frames: {n_frames}, detected frames: {n_detected}")
    st.subheader("Top-3 predictions")
    for w, p in top3:
        st.write(f"{w} ({bangla_label(w)}): {p:.1%}")
        st.progress(min(max(p, 0.0), 1.0))


if __name__ == "__main__":
    main()
