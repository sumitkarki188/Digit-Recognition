const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const predictedDigit = document.getElementById('predictedDigit');
const confidence = document.getElementById('confidence');
const probabilityBars = document.getElementById('probabilityBars');

let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Canvas setup
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = 'black';
ctx.lineWidth = 15;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

// Drawing functions
function startDrawing(e) {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX || e.touches[0].clientX) - rect.left;
    const y = (e.clientY || e.touches[0].clientY) - rect.top;
    lastX = x;
    lastY = y;
}

function draw(e) {
    if (!isDrawing) return;
    
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX || e.touches[0].clientX) - rect.left;
    const y = (e.clientY || e.touches[0].clientY) - rect.top;
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    
    lastX = x;
    lastY = y;
}

function stopDrawing() {
    isDrawing = false;
}

// Event listeners for mouse
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Event listeners for touch
canvas.addEventListener('touchstart', startDrawing);
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('touchend', stopDrawing);

// Clear button
clearBtn.addEventListener('click', () => {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    predictedDigit.textContent = '?';
    confidence.textContent = 'Draw a digit to predict';
    probabilityBars.innerHTML = '';
});

// Predict button
predictBtn.addEventListener('click', async () => {
    try {
        // Show loading state
        predictedDigit.textContent = '...';
        confidence.textContent = 'Analyzing...';
        predictBtn.disabled = true;
        
        // Get canvas data as base64
        const imageData = canvas.toDataURL('image/png');
        
        // Send to backend
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Display result
            predictedDigit.textContent = result.digit;
            confidence.textContent = `Confidence: ${result.confidence.toFixed(2)}%`;
            
            // Display probability bars
            displayProbabilities(result.probabilities);
        } else {
            predictedDigit.textContent = '❌';
            confidence.textContent = 'Error: ' + result.error;
        }
    } catch (error) {
        predictedDigit.textContent = '❌';
        confidence.textContent = 'Error connecting to server';
        console.error('Error:', error);
    } finally {
        predictBtn.disabled = false;
    }
});

function displayProbabilities(probabilities) {
    probabilityBars.innerHTML = '';
    
    for (let i = 0; i < 10; i++) {
        const prob = probabilities[i.toString()];
        const barContainer = document.createElement('div');
        barContainer.className = 'prob-bar-container';
        
        const label = document.createElement('span');
        label.className = 'prob-label';
        label.textContent = i;
        
        const barWrapper = document.createElement('div');
        barWrapper.className = 'prob-bar-wrapper';
        
        const bar = document.createElement('div');
        bar.className = 'prob-bar';
        bar.style.width = prob + '%';
        
        const value = document.createElement('span');
        value.className = 'prob-value';
        value.textContent = prob.toFixed(1) + '%';
        
        barWrapper.appendChild(bar);
        barContainer.appendChild(label);
        barContainer.appendChild(barWrapper);
        barContainer.appendChild(value);
        
        probabilityBars.appendChild(barContainer);
    }
}
