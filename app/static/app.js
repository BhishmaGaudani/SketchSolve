// SketchSolve frontend: draw on a canvas, send it to /predict, show the result.

const canvas = document.getElementById("draw");
const ctx = canvas.getContext("2d");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");

const solveButton = document.getElementById("solve");
const resultBox = document.getElementById("result");
const symbolsBox = document.getElementById("symbols");
const answerBox = document.getElementById("answer");
const lowNote = document.getElementById("low-note");

const LINE_WIDTH = 14; // in canvas pixels; the server normalizes stroke width anyway

// Every stroke is kept as a list of points, so Undo can redraw without the last one.
let strokes = [];
let current = null;

function clearCanvas() {
  ctx.fillStyle = "#fff"; // solid white, so the PNG has no transparent pixels
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function drawStroke(points) {
  ctx.strokeStyle = "#111";
  ctx.lineWidth = LINE_WIDTH;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  // A single tap still draws a dot (important for the dots of ÷).
  if (points.length === 1) ctx.lineTo(points[0].x + 0.1, points[0].y);
  for (const p of points.slice(1)) ctx.lineTo(p.x, p.y);
  ctx.stroke();
}

function redraw() {
  clearCanvas();
  strokes.forEach(drawStroke);
}

// The canvas is 1200x400 internally but shown at whatever size fits the page,
// so convert mouse/touch coordinates from screen pixels to canvas pixels.
function toCanvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * (canvas.width / rect.width),
    y: (event.clientY - rect.top) * (canvas.height / rect.height),
  };
}

// Pointer events cover mouse, pen and touch with one set of handlers.
canvas.addEventListener("pointerdown", (event) => {
  canvas.setPointerCapture(event.pointerId);
  current = [toCanvasPoint(event)];
  strokes.push(current);
  clearOverlay(); // old boxes no longer match the drawing
  drawStroke(current);
});

canvas.addEventListener("pointermove", (event) => {
  if (!current) return;
  current.push(toCanvasPoint(event));
  drawStroke(current.slice(-2)); // draw just the newest segment
});

function endStroke() { current = null; }
canvas.addEventListener("pointerup", endStroke);
canvas.addEventListener("pointercancel", endStroke);

function clearOverlay() {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
}

function drawBoxes(symbols) {
  clearOverlay();
  overlayCtx.lineWidth = 3;
  for (const s of symbols) {
    const [x, y, w, h] = s.box;
    overlayCtx.strokeStyle = s.low_confidence ? "#d9a300" : "rgba(47, 91, 211, 0.6)";
    overlayCtx.strokeRect(x - 6, y - 6, w + 12, h + 12);
  }
}

function showResult(data) {
  resultBox.hidden = false;
  symbolsBox.replaceChildren();

  for (const s of data.symbols || []) {
    const chip = document.createElement("div");
    chip.className = "chip" + (s.low_confidence ? " low" : "");
    chip.title = `Confidence ${(s.confidence * 100).toFixed(1)}%`;
    chip.textContent = s.symbol === "-" ? "−" : s.symbol; // nicer minus sign
    const conf = document.createElement("small");
    conf.textContent = `${Math.round(s.confidence * 100)}%`;
    chip.appendChild(conf);
    symbolsBox.appendChild(chip);
  }
  lowNote.hidden = !(data.symbols || []).some((s) => s.low_confidence);
  drawBoxes(data.symbols || []);

  // textContent (not innerHTML) so nothing from the server is run as HTML.
  if (data.error) {
    answerBox.textContent = data.error;
    answerBox.className = "answer error";
  } else {
    answerBox.textContent = data.answer;
    answerBox.className = "answer";
  }
}

async function solve() {
  solveButton.disabled = true;
  solveButton.textContent = "Solving…";
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: canvas.toDataURL("image/png") }),
    });
    const data = await response.json();
    showResult(response.ok ? data : { symbols: [], error: data.error || "Request failed." });
  } catch (err) {
    showResult({ symbols: [], error: "Couldn't reach the server." });
  } finally {
    solveButton.disabled = false;
    solveButton.textContent = "Solve";
  }
}

document.getElementById("clear").addEventListener("click", () => {
  strokes = [];
  redraw();
  clearOverlay();
  resultBox.hidden = true;
});

function undo() {
  strokes.pop();
  redraw();
  clearOverlay();
}

document.getElementById("undo").addEventListener("click", undo);
solveButton.addEventListener("click", solve);
document.addEventListener("keydown", (event) => {
  if (event.key === "Enter") solve();
  if ((event.ctrlKey || event.metaKey) && event.key === "z") undo();
});

clearCanvas();
