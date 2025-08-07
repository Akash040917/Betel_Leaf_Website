let model;

async function loadModel() {
    try {
        model = await tf.loadLayersModel('model/model.json');
        console.log("✅ Model loaded successfully!");
        document.getElementById("status").innerText = "✅ Model loaded successfully!";
    } catch (err) {
        console.error("❌ Failed to load the model:", err);
        document.getElementById("status").innerText = "❌ Failed to load the model.";
    }
}

loadModel();

function predict() {
    const img = document.getElementById('preview');
    if (!model) {
        alert("Model not loaded yet. Please wait.");
        return;
    }

    const tensor = tf.browser.fromPixels(img)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .div(tf.scalar(255.0))
        .expandDims();

    const prediction = model.predict(tensor);
    prediction.array().then(probabilities => {
        const classes = ['Anthracnose Green', 'BacterialLeafSpot Green', 'Healthy Green', 'Healthy Red'];
        const top = probabilities[0].map((p, i) => ({ className: classes[i], prob: p }))
                                   .sort((a, b) => b.prob - a.prob)[0];
        document.getElementById('result').innerText = `🟢 Prediction: ${top.className} (${(top.prob * 100).toFixed(2)}%)`;
    });
}

function previewImage(event) {
    const reader = new FileReader();
    reader.onload = function () {
        const output = document.getElementById('preview');
        output.src = reader.result;
        output.style.display = 'block';
    };
    reader.readAsDataURL(event.target.files[0]);
}

