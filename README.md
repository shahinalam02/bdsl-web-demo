# BdSL Recognition — Web Demo (Streamlit)

Small web app: upload a short video of one Bangla sign, the Transformer
model predicts the word (English + Bangla + confidence + top-3).

- Model: Transformer, 69.34% test accuracy, signer-independent split
  (test signers U4+U8, never seen in training), 10 BdSLW60 words,
  1,503 clips, 18 signers. Latency ~44 ms/window.
- `streamlit_app.py` — the app (video upload, no live webcam).
- `models.py`, `features.py` — model + MediaPipe Tasks landmark code.
- `pose_landmarker_lite.task`, `hand_landmarker.task` — MediaPipe models.
- `runs/run10/transformer_best.pt` — trained checkpoint (default).

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Opens at http://localhost:8501 — upload an .mp4/.mov/.avi clip of one sign.

## Deploy free (Streamlit Community Cloud)

1. Create a GitHub repo (e.g. `bdsl-web-demo`) and upload ALL files in
   this folder (including `runs/run10/transformer_best.pt`).
2. Go to https://share.streamlit.io → **New app** → pick the repo,
   branch `main`, main file `streamlit_app.py` → **Deploy**.
3. First build takes a few minutes (torch + mediapipe install).
   You get a public link like
   `https://bdsl-web-demo.streamlit.app` — anyone can open it and
   upload a sign video.

Notes:
- The app sleeps after ~inactivity on the free tier and wakes on visit.
- Video upload only (no live webcam in the browser version).
- `packages.txt` installs the system libs OpenCV/MediaPipe need.
