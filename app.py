from flask import Flask, render_template, request, jsonify
import keras
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

print("="*60)
print("Loading trained CNN model...")
try:
    model = keras.models.load_model('mnist_cnn_model.h5')
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error: {e}")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    
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
        
        # Preprocess
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
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("Server: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
