from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

# Load model once at startup
print("="*60)
print("Loading CNN model...")

model = None
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    
    # Load the model
    model = tf.keras.models.load_model('mnist_model.keras')
    print("✓ Model loaded successfully!")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    print("="*60)
    
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("="*60)
    import traceback
    traceback.print_exc()
    model = None

@app.route('/')
def index():
    """Render main application page"""
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
    """Handle prediction requests"""
    
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
                return jsonify({
                    'success': False,
                    'error': 'No file selected'
                }), 400
            
            img = Image.open(file.stream).convert('L')
            print("File uploaded")
        
        # Handle canvas drawing (base64)
        elif request.json and 'image' in request.json:
            img_data = request.json['image']
            
            # Remove data URL prefix
            if ',' in img_data:
                img_data = img_data.split(',')[1]
            
            # Decode base64
            img_bytes = base64.b64decode(img_data)
            img = Image.open(io.BytesIO(img_bytes)).convert('L')
            print("Canvas drawing received")
        
        else:
            return jsonify({
                'success': False,
                'error': 'No image data received'
            }), 400
        
        # Preprocess image
        # Resize to 28x28
        img_array = np.array(img.resize((28, 28), Image.LANCZOS))
        
        # Invert colors if white background (MNIST has white digits on black bg)
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
            print("Image colors inverted")
        
        # Normalize pixel values to 0-1
        img_array = img_array.astype('float32') / 255.0
        
        # Reshape to match model input: (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        
        # Get predicted digit and confidence
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]) * 100)
        
        # Get all probabilities
        all_predictions = {
            str(i): float(predictions[0][i] * 100) 
            for i in range(10)
        }
        
        print(f"Predicted: {predicted_digit} (Confidence: {confidence:.2f}%)")
        
        return jsonify({
            'success': True,
            'digit': predicted_digit,
            'confidence': round(confidence, 2),
            'probabilities': all_predictions
        })
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get('PORT', 5000))
    
    print("\n" + "="*60)
    print("Starting Flask Application")
    print("="*60)
    print(f"Server running on port: {port}")
    print("Press CTRL+C to quit")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=False)
