// ----------------------- FINAL SCRIPT.JS --------------------------

let model;
const classNames = [
  "Anthracnose_Green",
  "BacterialLeafSpot_Green",
  "Healthy_Green",
  "Healthy_Red"
];

// Load the TensorFlow.js converted fine-tuned model
async function loadModel() {
  document.getElementById("liveResult").innerText = "Loading Model...";
  model = await tf.loadLayersModel("tfjs_model1/model.json");   // <== do not change folder name
  document.getElementById("liveResult").innerText = "Model Loaded ✅";
  startWebcam();
}

// ------------------ REALTIME WEBCAM ------------------------- //
async function startWebcam() {
  const vid = document.getElementById("webcam");
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  vid.srcObject = stream;
  vid.addEventListener("loadeddata", predictLive);
}

async function predictLive() {
  const vid = document.getElementById("webcam");
  if (!model) return requestAnimationFrame(predictLive);

  const t = tf.browser.fromPixels(vid).resizeNearestNeighbor([224,224]).toFloat().div(255).expandDims();
  const p = await model.predict(t).data();
  const result = classNames[p.indexOf(Math.max(...p))];
  document.getElementById("liveResult").innerText = "Live Prediction: " + result;
  requestAnimationFrame(predictLive);
}

// -------------------- UPLOAD PREDICTION ---------------------- //
document.getElementById("uploadInput").addEventListener("change", function (e) {
  const file = e.target.files[0];
  if (!file) return;
  const img = document.getElementById("previewImg");
  img.src = URL.createObjectURL(file);
  img.style.display = "block";

  img.onload = async function () {
    if (!model) { alert("Model not yet loaded"); return; }
    const t = tf.browser.fromPixels(img).resizeNearestNeighbor([224,224]).toFloat().div(255).expandDims();
    const p = await model.predict(t).data();
    const result = classNames[p.indexOf(Math.max(...p))];
    document.getElementById("uploadResult").innerText = "Upload Prediction: " + result;
  }
});

// ------------ START -------------
loadModel();

