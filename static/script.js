const video = document.getElementById('video');
const captureBtn = document.getElementById('capture-btn');
const uploadSection = document.getElementById('upload-section');
const fileInput = document.getElementById('file-input');
const result = document.getElementById('result');

// Enable webcam
function enableWebcam() {
    uploadSection.style.display = 'none';
    video.style.display = 'block';
    captureBtn.style.display = 'inline-block';

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
        });
}

// Enable upload input
function enableUpload() {
    video.style.display = 'none';
    captureBtn.style.display = 'none';
    uploadSection.style.display = 'block';

    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }
}

// Capture from webcam and send
function captureAndSend() {
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, 224, 224);
    const imageData = canvas.toDataURL('image/jpeg');
    sendToServer(imageData);
}

// Upload and send
function uploadAndSend() {
    const file = fileInput.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onloadend = function () {
        sendToServer(reader.result);
    };
    reader.readAsDataURL(file);
}

// Send image to server
function sendToServer(imageData) {
    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageData })
    })
    .then(res => res.json())
    .then(data => {
        result.textContent = "Prediction: " + data.prediction + " Taka";
        const utterance = new SpeechSynthesisUtterance(data.prediction + " Taka");
        window.speechSynthesis.speak(utterance);
    })
    .catch(err => {
        result.textContent = "Error: " + err.message;
    });
}
