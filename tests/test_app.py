import base64

import cv2
import numpy as np
import pytest

from app.server import create_app


@pytest.fixture(scope="module")
def client():
    return create_app().test_client()  # loads the real model once for all tests


def to_data_url(image: np.ndarray) -> str:
    ok, png = cv2.imencode(".png", image)
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()


def blank_canvas():
    return np.full((400, 1200), 255, dtype=np.uint8)  # white, like the frontend


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"SketchSolve" in response.data


def test_empty_canvas_gets_friendly_message(client):
    response = client.post("/predict", json={"image": to_data_url(blank_canvas())})
    assert response.status_code == 200
    data = response.get_json()
    assert data["symbols"] == []
    assert data["answer"] is None
    assert "empty" in data["error"]


def test_one_plus_one(client):
    img = blank_canvas()
    cv2.line(img, (150, 100), (150, 300), 0, 14)                  # 1
    cv2.line(img, (330, 200), (490, 200), 0, 14)                  # + horizontal
    cv2.line(img, (410, 120), (410, 280), 0, 14)                  # + vertical
    cv2.line(img, (650, 100), (650, 300), 0, 14)                  # 1
    data = client.post("/predict", json={"image": to_data_url(img)}).get_json()
    assert [s["symbol"] for s in data["symbols"]] == ["1", "+", "1"]
    assert data["expression"] == "1+1"
    assert data["answer"] == "2"
    assert all(0 <= s["confidence"] <= 1 for s in data["symbols"])


def test_transparent_png_is_read_as_white_background(client):
    rgba = np.zeros((400, 1200, 4), dtype=np.uint8)               # fully transparent
    cv2.line(rgba, (150, 100), (150, 300), (0, 0, 0, 255), 14)    # opaque black "1"
    data = client.post("/predict", json={"image": to_data_url(rgba)}).get_json()
    assert data["expression"] == "1"


@pytest.mark.parametrize("body", [
    None,                            # not JSON
    {},                              # no image field
    {"image": "not base64!!"},
    {"image": base64.b64encode(b"hello").decode()},  # base64, but not an image
])
def test_bad_requests_get_400(client, body):
    response = client.post("/predict", json=body) if body is not None \
        else client.post("/predict", data="plain text")
    assert response.status_code == 400
    assert "error" in response.get_json()
