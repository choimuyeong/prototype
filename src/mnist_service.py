from __future__ import annotations

from datetime import datetime
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import onnxruntime as ort
from PIL import Image
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "mnist-12.onnx"

MODEL_URLS = [
    "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx",
    "https://raw.githubusercontent.com/onnx/models/main/validated/vision/classification/mnist/model/mnist-12.onnx",
    "https://github.com/onnx/models/raw/main/vision/classification/mnist/model/mnist-12.onnx",
]


def _download_model() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for url in MODEL_URLS:
        try:
            urlretrieve(url, MODEL_PATH)
            return
        except Exception as exc:  # pragma: no cover
            last_error = exc
    raise RuntimeError("MNIST ONNX 모델 다운로드에 실패했습니다.") from last_error


def ensure_model_exists() -> Path:
    if not MODEL_PATH.exists():
        _download_model()
    return MODEL_PATH


def get_model_info() -> dict:
    exists = MODEL_PATH.exists()
    size_bytes = MODEL_PATH.stat().st_size if exists else 0
    return {
        "path": str(MODEL_PATH),
        "exists": exists,
        "size_bytes": size_bytes,
        "size_kb": round(size_bytes / 1024, 2) if exists else 0.0,
        "last_modified": datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).isoformat(timespec="seconds") if exists else None,
    }


def redownload_model() -> Path:
    MODEL_PATH.unlink(missing_ok=True)
    _download_model()
    return MODEL_PATH


def reload_session() -> ort.InferenceSession:
    get_session.clear()
    return get_session()


def preprocess_canvas(image_rgba: np.ndarray) -> np.ndarray:
    gray = Image.fromarray(image_rgba.astype(np.uint8)).convert("L")
    arr = np.array(gray).astype(np.float32) / 255.0

    ys, xs = np.where(arr > 0.05)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((28, 28), dtype=np.float32)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cropped = arr[y_min : y_max + 1, x_min : x_max + 1]

    digit = Image.fromarray((cropped * 255).astype(np.uint8)).resize((20, 20), Image.Resampling.BILINEAR)
    canvas = np.zeros((28, 28), dtype=np.float32)
    canvas[4:24, 4:24] = np.array(digit).astype(np.float32) / 255.0
    return canvas


def to_model_input(img_28x28: np.ndarray) -> np.ndarray:
    return img_28x28.reshape(1, 1, 28, 28).astype(np.float32)


@st.cache_resource(show_spinner=False)
def get_session() -> ort.InferenceSession:
    ensure_model_exists()
    return ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


def predict(img_28x28: np.ndarray) -> tuple[int, np.ndarray]:
    x = to_model_input(img_28x28)
    session = get_session()
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    logits = session.run([output_name], {input_name: x})[0]

    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    pred = int(np.argmax(probs, axis=1)[0])
    return pred, probs[0]
