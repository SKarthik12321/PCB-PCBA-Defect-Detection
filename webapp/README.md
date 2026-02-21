# PCB Defect Detection Webapp

This directory contains a small FastAPI backend and a React (CDN) frontend for testing image upload and webcam-based PCB defect detection.

How it works:
- Backend: `webapp/backend/app.py` - provides `/api/detect-upload` and `/api/detect-frame` endpoints, and serves the frontend static files.
- Frontend: `webapp/frontend/index.html`, `app.js`, `style.css` - simple React UI that can upload images or stream webcam frames to the backend.

Quick start (from repository root):

1. Create and activate the Python environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
pip install fastapi uvicorn[standard] python-multipart opencv-python
```

3. Run the backend server:

```bash
python webapp/backend/app.py
```

4. Open `http://localhost:8000` in your browser.

Notes:
- For accurate PCB detections, place your trained `pcb_model.pt` or `pcb_model.onnx` in the repository `models/` directory. If no model is present, the server will fall back to `yolov8n.pt` (downloaded automatically by Ultralytics) which is only for demo.
- The webcam flow posts JPEG frames periodically; this is a simple approach that requires network bandwidth and is not highly optimized for low-latency streaming.
