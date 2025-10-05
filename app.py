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
        print("=" * 50)
        print("INITIALIZING DIGIT RECOGNITION APP")
        print("=" * 50)
        print("Loading pre-trained model...")
        
        model_path = 'digit_model.h5'
        
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found at: {model_path}")
            print(f"Current directory: {os.getcwd()}")
            print(f"Files in directory: {os.listdir('.')}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = keras.models.load_model(model_path)
        print("Model loaded successfully!")
        
        # Test the model
        test_input = np.zeros((1, 28, 28, 1))
        _ = model.predict(test_input, verbose=0)
        print("Model test prediction successful!")
        print("=" * 50)
        print("Model ready! Application started.")
        print("=" * 50)
        
        return model
    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e

# Load model when the module is imported (before Gunicorn forks workers)
print("Starting application initialization...")
load_model_once()
print("Application initialization complete!")

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
            new_w = int(20 * w / h) if w > 0 else 20
        else:
            new_w = 20
            new_h = int(20 * h / w) if h > 0 else 20
        
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
        import traceback
        traceback.print_exc()
        raise e

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            print("ERROR: Model is None when trying to predict")
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please contact administrator.'
            })
        
        # Get image data from request
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({
                'success': False,
                'error': 'No image data received'
            })
        
        # Preprocess the image
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        predicted_digit = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_digit]) * 100
        
        # Get all probabilities
        probabilities = {str(i): float(prediction[0][i] * 100) for i in range(10)}
        
        print(f"Prediction successful: digit={predicted_digit}, confidence={confidence:.2f}%")
        
        return jsonify({
            'success': True,
            'digit': int(predicted_digit),
            'confidence': round(confidence, 2),
            'probabilities': probabilities
        })
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    # This block is only for local development
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
