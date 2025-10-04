from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

# Load model with better error handling
print("="*60)
print("Loading trained CNN model...")

model = None
try:
    # Try loading .keras format first (Keras 3)
    from tensorflow.keras.models import load_model
    try:
        model = load_model('mnist_cnn_model.keras')
        print("✓ Model loaded from mnist_cnn_model.keras")
    except:
        # Fall back to .h5 format
        model = load_model('mnist_cnn_model.h5', compile=False)
        # Recompile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✓ Model loaded from mnist_cnn_model.h5 and recompiled")
    
    print(f"Model input shape: {model.input_shape}")
    print("="*60)
    
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("="*60)
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please contact administrator.'
        }), 500
    
    try:
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'}), 400
            img = Image.open(file.stream).convert('L')
            print("File uploaded")
        
        # Handle canvas drawing
        elif request.json and 'image' in request.json:
            img_data = request.json['image']
            if ',' in img_data:
                img_data = img_data.split(',')[1]
            img_bytes = base64.b64decode(img_data)
            img = Image.open(io.BytesIO(img_bytes)).convert('L')
            print("Canvas drawing")
        else:
            return jsonify({'success': False, 'error': 'No image data'}), 400
        
        # Preprocess image
        img_array = np.array(img.resize((28, 28), Image.LANCZOS))
        
        # Invert if white background
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
        
        # Normalize and reshape
        img_array = img_array.astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]) * 100)
        
        all_predictions = {str(i): float(predictions[0][i] * 100) for i in range(10)}
        
        print(f"Predicted: {predicted_digit} ({confidence:.2f}%)")
        
        return jsonify({
            'success': True,
            'digit': predicted_digit,
            'confidence': round(confidence, 2),
            'probabilities': all_predictions
        })
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
