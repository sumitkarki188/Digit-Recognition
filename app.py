from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
from tensorflow import keras
import os

app = Flask(__name__)

# Global variable to store the loaded model
model = None

def load_model_once():
    """Load the pre-trained model once when the application starts"""
    global model
    
    try:
        print("Loading pre-trained model...")
        model_path = 'digit_model.h5'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = keras.models.load_model(model_path)
        print("Model loaded successfully!")
        
        # Test the model
        test_input = np.zeros((1, 28, 28, 1))
        _ = model.predict(test_input, verbose=0)
        print("Model test prediction successful!")
        
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise e

def preprocess_image(image_data):
    """Enhanced preprocessing for better accuracy"""
    try:
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # Apply Gaussian blur to reduce noise
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Apply adaptive thresholding
        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2 * padding)
            h = min(img.shape[0] - y, h + 2 * padding)
            
            digit = img[y:y+h, x:x+w]
        else:
            digit = img
        
        # Resize maintaining aspect ratio
        h, w = digit.shape
        if h > w:
            new_h = 20
            new_w = int(20 * w / h)
        else:
            new_w = 20
            new_h = int(20 * h / w)
        
        digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Center in 28x28
        final_img = np.zeros((28, 28), dtype=np.uint8)
        y_offset = (28 - new_h) // 2
        x_offset = (28 - new_w) // 2
        final_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_resized
        
        # Normalize
        final_img = final_img.astype('float32') / 255.0
        final_img = final_img.reshape(1, 28, 28, 1)
        
        return final_img
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise e

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please refresh the page.'
            })
        
        # Get image data from request
        data = request.get_json()
        image_data = data['image']
        
        # Preprocess the image
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        predicted_digit = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_digit]) * 100
        
        # Get all probabilities
        probabilities = {str(i): float(prediction[0][i] * 100) for i in range(10)}
        
        return jsonify({
            'success': True,
            'digit': int(predicted_digit),
            'confidence': round(confidence, 2),
            'probabilities': probabilities
        })
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    # Load the model once when the app starts
    print("=" * 50)
    print("INITIALIZING DIGIT RECOGNITION APP")
    print("=" * 50)
    
    try:
        load_model_once()
        print("=" * 50)
        print("Model ready! Starting Flask server...")
        print("=" * 50)
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        print("Application will start but predictions will fail.")
    
    # Get port from environment variable
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
