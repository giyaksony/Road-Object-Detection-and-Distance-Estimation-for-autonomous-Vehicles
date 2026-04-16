// script.js
const uploadForm = document.getElementById('upload-form');
const videoFileInput = document.getElementById('video-file');
const analyzeBtn = document.getElementById('analyze-btn');
const liveBtn = document.getElementById('live-btn');
const loadingDiv = document.getElementById('loading');
const uploadSection = document.getElementById('upload-section');
const resultSection = document.getElementById('result-section');
const videoStreamImg = document.getElementById('video-stream');

// Enable analyze button only when file is selected
videoFileInput.addEventListener('change', () => {
    analyzeBtn.disabled = videoFileInput.files.length === 0;
});

// Handle Video Upload
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    startProcessing('/upload', true);
});

// Handle Live Camera
liveBtn.addEventListener('click', () => {
    startProcessing('/upload', false);
});

async function startProcessing(url, isUpload) {
    loadingDiv.classList.remove('hidden');

    let fetchOptions = { method: 'POST' };

    if (isUpload) {
        const formData = new FormData();
        formData.append('video', videoFileInput.files[0]);
        fetchOptions.body = formData;
    }

    try {
        const response = await fetch(url, fetchOptions);
        const data = await response.json();

        if (response.ok) {
            // Set the stream source to the URL returned by Flask
            videoStreamImg.src = data.video_url;

            uploadSection.classList.add('hidden');
            resultSection.classList.remove('hidden');
            loadingDiv.classList.add('hidden');
        } else {
            alert("Error: " + data.error);
            loadingDiv.classList.add('hidden');
        }
    } catch (err) {
        alert("Failed to connect to the AI server.");
        loadingDiv.classList.add('hidden');
    }
}

// Reset Logic
document.getElementById('new-video-btn').addEventListener('click', () => {
    videoStreamImg.src = ""; // Force stop the stream
    resultSection.classList.add('hidden');
    uploadSection.classList.remove('hidden');
    uploadForm.reset();
    analyzeBtn.disabled = true;
});