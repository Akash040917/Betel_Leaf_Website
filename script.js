let model;

const MODEL_URL = 'model/model.json';

async function loadModel() {
  try {
    model = await tf.loadLayersModel(MODEL_URL);
    console.log("✅ Model loaded successfully.");
  } catch (error) {
    console.error("❌ Failed to load the model:", error);
  }
}

loadModel();

document.getElementById("imageUpload").addEventListener("change", function (event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function (e) {
    const img = document.getElementById("preview");
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
});

async function predict() {
  if (!model) {
    alert("Model not loaded yet. Please wait...");
    return;
  }

  const imageElement = document.getElementById("preview");

  if (!imageElement.src || imageElement.src === window.location.href) {
    alert("Please upload an image first.");
    return;
  }

  const tensor = tf.browser.fromPixels(imageElement)
    .resizeNearestNeighbor([224, 224])
    .toFloat()
    .div(tf.scalar(255.0))
    .expandDims();

  const prediction = model.predict(tensor);
  const predictedClass = prediction.argMax(-1);
  const classIndex = (await predictedClass.data())[0];

  const classNames = ["Class A", "Class B", "Class C"]; // Replace with your actual class labels
  const label = classNames[classIndex] || `Class ${classIndex}`;

  document.getElementById("prediction").textContent = `Prediction: ${label}`;
}


