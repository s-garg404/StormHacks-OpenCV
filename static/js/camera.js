const cameraButton = document.getElementById("camera-button");
const video = document.getElementById("camera-stream");
const canvas = document.createElement("canvas");
const resultDiv = document.getElementById("prediction-result");

let stream = null;

// Start the camera
cameraButton.addEventListener("click", async () => {
  console.log("Camera started");
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.style.display = "block";

    // Automatically start classifying every few seconds
    setInterval(captureAndSendFrame, 5000); // every 5s
  } catch (err) {
    console.error("Camera error:", err);
    alert("Camera access failed: " + err.message);
  }
});

async function captureAndSendFrame() {
  if (!video.srcObject) return;

  // Draw video frame onto canvas
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);

  // Convert to blob and send to Flask
  canvas.toBlob(async (blob) => {
    const formData = new FormData();
    formData.append("image", blob, "frame.jpg");

    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    console.log("Prediction:", data.prediction);

    resultDiv.innerHTML = `<h2>Prediction: ${data.prediction}</h2>`;
    highlightCategory(data.prediction);
  }, "image/jpeg");
}

function highlightCategory(categoryName) {
  document.querySelectorAll('.category').forEach(cat => cat.classList.remove('highlight'));
  const el = document.getElementById(categoryName.toLowerCase());
  if (el) el.classList.add('highlight');
}
