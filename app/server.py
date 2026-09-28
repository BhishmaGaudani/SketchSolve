"""Flask backend for SketchSolve.

Run:  python -m app.server        (http://localhost:8080)

Set SKETCHSOLVE_DEBUG=1 to save a debug image of every request to debug/.
"""
import base64
import binascii
import os
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from src.config import ROOT
from src.pipeline import Recognizer

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # a canvas PNG is ~20-100 KB; reject anything huge
MAX_SIDE = 3000                     # pixels; bigger images are rejected


class BadImage(Exception):
    pass


def decode_image(data_url: str) -> np.ndarray:
    """Turn the canvas's "data:image/png;base64,..." string into a grayscale array."""
    if not isinstance(data_url, str) or not data_url:
        raise BadImage("Missing 'image' field.")
    encoded = data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise BadImage("'image' is not valid base64.")

    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise BadImage("'image' is not a readable PNG or JPEG.")
    if max(image.shape[:2]) > MAX_SIDE:
        raise BadImage(f"Image is larger than {MAX_SIDE}px.")

    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        # Transparent pixels (alpha 0) count as background: put the image on white.
        alpha = image[:, :, 3:4].astype(np.float32) / 255
        rgb = image[:, :, :3] * alpha + 255 * (1 - alpha)
        image = rgb.astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    recognizer = Recognizer()  # loads the model once, not on every request
    save_debug = os.environ.get("SKETCHSOLVE_DEBUG") == "1"

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict")
    def predict():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify(error="Send JSON like {\"image\": \"data:image/png;base64,...\"}."), 400
        try:
            gray = decode_image(body.get("image"))
        except BadImage as e:
            return jsonify(error=str(e)), 400

        debug_path = ROOT / "debug" / f"request_{time.time_ns()}.png" if save_debug else None
        result = recognizer.recognize(gray, debug_path=debug_path)
        # 200 even when result.error is set: the request worked, the drawing
        # just couldn't be solved. The frontend shows the message.
        return jsonify(asdict(result))

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="Image is too large."), 413

    @app.errorhandler(500)
    def server_error(_):
        # Don't leak internal details (stack traces) to the client.
        return jsonify(error="Something went wrong on the server."), 500

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=int(os.environ.get("PORT", 8080)), debug=False)
