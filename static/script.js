// Canvas setup
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Canvas settings
ctx.lineWidth = 20;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.strokeStyle = 'black';
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Tab switching
function showTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    if (tabName === 'draw') {
        document.getElementById('draw-tab').classList.add('active');
        document.querySelectorAll('.tab-btn')[0].classList.add('active');
    } else {
        document.getElementById('upload-tab').classList.add('active');
        document.querySelectorAll('.tab-btn')[1].classList.add('active');
    }
    
    hideResults();
}

// Drawing functions
function startDrawing(e) {
    isDrawing = true;
    [lastX, lastY] = getMousePos(e);
}

function draw(e) {
    if (!isDrawing) return;
    
    const [x, y] = getMousePos(e);
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    
    [lastX, lastY] = [x, y];
}

function stopDrawing() {
    isDrawing = false;
}

function getMousePos(e) {
    const rect = canvas.getBoundingClientRect();
    const touch = e.touches ? e.touches[0] : e;
    return [
        touch.clientX - rect.left,
        touch.clientY - rect.top
    ];
}

// Mouse events
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Touch events
canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    startDrawing(e);
});
canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    draw(e);
});
canvas.addEventListener('touchend', stopDrawing);

// Clear canvas
document.getElementById('clear-btn').addEventListener('click', () => {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    hideResults();
});

// File upload preview
const fileInput = document.getElementById('file-input');
const previewContainer = document.getElementById('preview-container');
const previewImage = document.getElementById('preview-image');

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewContainer.style.display = 'block';
        };
        reader.readAsDataURL(file);
        hideResults();
    }
});

// Predict from canvas
document.getElementById('predict-draw-btn').addEventListener('click', async () => {
    const imageData = canvas.toDataURL('image/png');
    await sendPrediction({ image: imageData });
});

// Predict from upload
document.getElementById('predict-upload-btn').addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        displayResults(data);
    } catch (error) {
        alert('Error: ' + error.message);
    }
});

// Send prediction request
async function sendPrediction(data) {
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        displayResults(result);
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

// Display results
function displayResults(data) {
    if (!data.success) {
        alert('Prediction failed: ' + data.error);
        return;
    }
    
    const resultSection = document.getElementById('result-section');
    const predictedDigit = document.getElementById('predicted-digit');
    const confidence = document.getElementById('confidence');
    const probabilityBars = document.getElementById('probability-bars');
    
    predictedDigit.textContent = data.digit;
    confidence.textContent = data.confidence + '%';
    
    // Create probability bars
    probabilityBars.innerHTML = '';
    for (let i = 0; i < 10; i++) {
        const prob = data.probabilities[i.toString()];
        const barHTML = `
            <div class="prob-bar">
                <div class="prob-label">${i}:</div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill" style="width: ${prob}%"></div>
                </div>
                <div class="prob-value">${prob.toFixed(1)}%</div>
            </div>
        `;
        probabilityBars.innerHTML += barHTML;
    }
    
    resultSection.style.display = 'block';
    resultSection.scrollIntoView({ behavior: 'smooth' });
}

function hideResults() {
    document.getElementById('result-section').style.display = 'none';
}
