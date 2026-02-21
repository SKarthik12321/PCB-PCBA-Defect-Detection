from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
from contextlib import asynccontextmanager
from PIL import Image
import shutil
import tempfile
from pathlib import Path
import io
import cv2
import json

import sys
from pathlib import Path as _Path
# Ensure repository root is on sys.path so `src` imports work when running this file directly
repo_root = _Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.model import PCBDetector


app = FastAPI(title="PCB Defect Detection Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-detection-meta"],
)

# Serve Vite React build from webapp/react-frontend/dist
# Run `cd webapp/react-frontend && npm run build` once to generate the dist folder
frontend_dir = Path(__file__).resolve().parents[1] / "react-frontend" / "dist"
if frontend_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dir / "assets")), name="assets")
else:
    # Fallback to old plain HTML frontend during development
    frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
    app.mount("/static", StaticFiles(directory=str(frontend_dir), html=True), name="frontend_static")


@app.get("/")
async def root_index():
    """Serve the frontend index page."""
    return FileResponse(frontend_dir / "index.html")


def get_model_info():
    """Find which model file will be loaded."""
    models_dir = repo_root / "models"
    candidates = [
        ("pcb_model.onnx", models_dir / "pcb_model.onnx"),
        ("pcb_model.pt",   models_dir / "pcb_model.pt"),
        ("yolov8n.pt",     None),  # fallback – no file check needed
    ]
    for name, path in candidates:
        if path is None or path.exists():
            has_pcb = name != "yolov8n.pt"
            return name, has_pcb
    return "yolov8n.pt", False


def get_detector() -> PCBDetector:
    """Load a trained model if available else fall back to a small public one."""
    model_name, _ = get_model_info()
    models_dir = repo_root / "models"

    if model_name == "yolov8n.pt":
        return PCBDetector(model_path="yolov8n.pt")

    candidate = models_dir / model_name
    try:
        return PCBDetector.load_trained(str(candidate))
    except Exception:
        return PCBDetector(model_path="yolov8n.pt")


DETECTOR = None
MODEL_NAME = "yolov8n.pt"
HAS_PCB_MODEL = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global DETECTOR, MODEL_NAME, HAS_PCB_MODEL
    MODEL_NAME, HAS_PCB_MODEL = get_model_info()
    DETECTOR = get_detector()
    print(f"✅ Model loaded: {MODEL_NAME} (PCB-specific: {HAS_PCB_MODEL})")
    yield


app.router.lifespan_context = lifespan


@app.get("/api/model-status")
async def model_status():
    """Return info about the currently loaded model."""
    return JSONResponse({
        "model_name": MODEL_NAME,
        "has_pcb_model": HAS_PCB_MODEL,
        "ready": DETECTOR is not None,
    })


def run_detection(img_path: Path):
    """Run detector on `img_path`. Returns (annotated_jpeg_bytes, detections_list)."""
    results = DETECTOR.predict(source=str(img_path), save=False, show=False)
    if not results:
        raise RuntimeError("No results from model")

    result = results[0]

    # Build detection metadata
    detections = []
    if result.boxes is not None:
        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf   = float(boxes.conf[i].item())
            cls_name = result.names.get(cls_id, str(cls_id))
            xyxy = boxes.xyxy[i].tolist()
            detections.append({
                "class_id":   cls_id,
                "class_name": cls_name,
                "confidence": round(conf, 4),
                "bbox": [round(v, 1) for v in xyxy],
            })

    # Annotated image
    img = result.plot()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ret, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return io.BytesIO(buf.tobytes()), detections


@app.post("/api/detect-upload")
async def detect_upload(file: UploadFile = File(...)):
    """Accept an uploaded image file and return annotated image with detection metadata header."""
    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        shutil.copyfileobj(file.file, tmp)

    try:
        # If OpenCV cannot read the format (e.g., AVIF), try PIL conversion to JPEG
        test = cv2.imread(str(tmp_path))
        if test is None:
            try:
                pil_img = Image.open(str(tmp_path)).convert('RGB')
                conv_tmp = tmp_path.with_suffix('.jpg')
                pil_img.save(str(conv_tmp), format='JPEG', quality=90)
                tmp_path.unlink()
                tmp_path = conv_tmp
            except Exception:
                pass

        img_bytes, detections = run_detection(tmp_path)
        img_bytes.seek(0)

        meta = json.dumps({"count": len(detections), "detections": detections})

        return Response(
            content=img_bytes.read(),
            media_type="image/jpeg",
            headers={"x-detection-meta": meta},
        )
    except Exception as ex:
        return JSONResponse({"error": "detection_failed", "detail": str(ex)}, status_code=500)
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass


@app.post("/api/detect-frame")
async def detect_frame(request: Request):
    """Accept raw image bytes (from webcam capture) and return annotated JPEG image."""
    body = await request.body()
    if not body:
        return JSONResponse({"error": "empty body"}, status_code=400)

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp.write(body)
        tmp_path = Path(tmp.name)

    try:
        img_bytes, detections = run_detection(tmp_path)
        img_bytes.seek(0)

        meta = json.dumps({"count": len(detections), "detections": detections})
        return Response(
            content=img_bytes.read(),
            media_type="image/jpeg",
            headers={"x-detection-meta": meta},
        )
    except Exception as ex:
        return JSONResponse({"error": "detection_failed", "detail": str(ex)}, status_code=500)
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run("webapp.backend.app:app", host="0.0.0.0", port=8000, reload=True)
