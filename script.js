let model;
let modelLoaded = false;

async function loadModel() {
  model = await tf.loadLayersModel('model/model.json');
  console.log("✅ Model loaded");
  modelLoaded = true;
}

loadModel();

document.getElementById('imageUpload').addEventListener('change', async function (event) {
  if (!modelLoaded) {
    alert("Model not loaded yet. Please wait.");
    return;
  }

  const file = event.target.files[0];
  const img = document.getElementById('preview');
  img.src = URL.createObjectURL(file);

  img.onload = async () => {
    const tensor = tf.browser
      .fromPixels(img)
      .resizeNearestNeighbor([224, 224])
      .toFloat()
      .div(255.0)
      .expandDims();

    const prediction = model.predict(tensor);
    const predictedClass = (await prediction.argMax(-1).data())[0];

    const classLabels = [
      "Anthracnose Green",
      "BacterialLeafSpot Green",
      "Healthy Green",
      "Healthy Red"
    ];

    document.getElementById('result').innerText =
      `Prediction: ${classLabels[predictedClass]}`;
  };
});
