const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

// Init canvas background to white
function initCanvas() {
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#000";
  ctx.lineWidth = 10;
  ctx.lineCap = "round";
}
initCanvas();

window.onload = () => {
  initCanvas();
};

let drawing = false;

canvas.addEventListener("mousedown", (e) => {
  drawing = true;
  ctx.beginPath();
  ctx.moveTo(e.offsetX, e.offsetY);
});

canvas.addEventListener("mousemove", (e) => {
  if (!drawing) return;
  ctx.lineTo(e.offsetX, e.offsetY);
  ctx.stroke();
});

canvas.addEventListener("mouseup", () => {
  drawing = false;
  ctx.closePath();
});

canvas.addEventListener("mouseleave", () => drawing = false);

// Touch events for mobile compatibility
canvas.addEventListener("touchstart", (e) => {
  e.preventDefault();
  const touch = e.touches[0];
  const rect = canvas.getBoundingClientRect();
  const x = touch.clientX - rect.left;
  const y = touch.clientY - rect.top;
  drawing = true;
  ctx.beginPath();
  ctx.moveTo(x, y);
});

canvas.addEventListener("touchmove", (e) => {
  e.preventDefault();
  if (!drawing) return;
  const touch = e.touches[0];
  const rect = canvas.getBoundingClientRect();
  const x = touch.clientX - rect.left;
  const y = touch.clientY - rect.top;
  ctx.lineTo(x, y);
  ctx.stroke();
});

canvas.addEventListener("touchend", (e) => {
  e.preventDefault();
  drawing = false;
  ctx.closePath();
});

function clearCanvas() {
  initCanvas();
}

function updateFileName() {
  const input = document.getElementById("fileInput");
  const label = document.getElementById("fileName");
  label.textContent = input.files[0]?.name || "Choose an image";
}

function sendCanvas() {
  canvas.toBlob((blob) => {
    const formData = new FormData();
    formData.append("image", blob, "canvas.png");

    fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData
    })
    .then(res => {
      if (!res.ok) {
        throw new Error(`Server responded with status ${res.status}`);
      }
      return res.json();
    })
    .then(data => {
      document.getElementById("result").innerText = data.prediction;
    })
    .catch(err => {
      console.error("Canvas prediction failed:", err);
      document.getElementById("result").innerText = "Prediction failed. Please try again.";
    });
  }, "image/png");
}

function sendFile() {
  const input = document.getElementById("fileInput");
  const file = input.files[0];

  if (!file) return;

  if (!file.type.startsWith("image/")) {
    alert("Please upload a valid image file.");
    return;
  }

  const formData = new FormData();
  formData.append("image", file);

  fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    body: formData
  })
  .then(res => {
    if (!res.ok) {
      throw new Error(`Server responded with status ${res.status}`);
    }
    return res.json();
  })
  .then(data => {
    document.getElementById("result").innerText = data.prediction;
  })
  .catch(err => {
    console.error("File prediction failed:", err);
    document.getElementById("result").innerText = "Prediction failed. Please try again.";
  });
}